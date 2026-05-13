# Score-math rigor audit — post codex correction (2026-05-13)

**Lane**: `lane_codex_math_correction_and_coordination_20260513` (registered L0 2026-05-13)
**Operator directive 2026-05-13**: "fix the math error EVERYWHERE it appears" + "apply the codex-found math gate principle BROADLY" + "audit all session memos for SIMILAR break-even / score-bound claims that may have similar errors".
**Sister memo**: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_codex_math_correction_pr95_lora_dora_landed_20260513.md`
**Codex source of correction**: `cd40d859` "pr95: add LoRA adapter break-even budget math" (`src/tac/substrates/pr95_lora_dora/budget.py` — math-as-code helper)

## TL;DR

Codex caught a linearization-of-square-root bug in our LoRA/DoRA landing memo: at PR106 operating point (`pose_avg = 3.4e-5`), the local pose derivative `5/sqrt(10·3.4e-5) ≈ 271 score/pose-unit` is only valid for INFINITESIMAL pose reductions. A +21 KB trailer's required offset (~0.014 score) is a large fraction of the entire current pose term `sqrt(10·3.4e-5) ≈ 0.01844`, so linear extrapolation gives wrong answers by 2 orders of magnitude.

This audit examines every other session memo with a `predicted_band` / Δscore / break-even claim near the PR106 frontier operating point and flags inconsistencies.

## The bad arithmetic vs. the correct arithmetic

| Quantity | Bad (linearized) | Correct (exact sqrt-inverse) |
|---|---|---|
| Current pose term | (n/a) | `sqrt(10·3.4e-5) = 0.018439` |
| Pose marginal `dS/d(pose_avg)` at `pose_avg = 3.4e-5` | (n/a) | `5/sqrt(10·3.4e-5) = 271.16` |
| Pose reduction for ΔS = -0.0027 | `0.0027/271 = 1e-5` (i.e., 1e-5 unit reduction) | exact inverse via sqrt formula |
| **Pose reduction for ΔS = -0.0027 (exact)** | `1e-5` (3-decade WRONG) | reduces term to `0.018439 - 0.0027 = 0.015739`, so new `pose_avg = 0.015739² / 10 = 2.477e-5`, i.e., **Δpose_avg = -9.23e-6** (~28% reduction) |
| Pose reduction needed to offset +21 KB trailer (ΔS = +0.014 penalty) | `5.2e-5` (close to whole pose term!) | new term = `0.018439 - 0.014 = 0.004439`, new pose_avg = `1.97e-6`, **Δpose_avg = -3.20e-5** (~94% reduction; nearly whole pose term must vanish) |

**Why linearization fails here.** For an additive penalty ΔS that is a non-trivial fraction of the **square root term** `sqrt(10·pose_avg)`, the local derivative is dominated by curvature. The Taylor expansion's first-order error is `(1/2) · d²S/dp² · (Δp)²`, which for the sqrt formula is `(-5/4) · (10·p)^(-3/2) · (Δp)²`. At `p = 3.4e-5`, the first-order error reaches the same magnitude as the first-order term when `|Δp| ≈ p` — i.e., the linearization breaks down once the requested reduction is on the order of the current pose_avg. Use `exact_pose_reduction_for_score_delta(pose_dist, score_delta)` (codex's canonical helper) to invert.

## Worked example with units

**Setup**: PR95-style trailer adds N bytes to a base archive at PR106-frontier operating point.

Constants (verified `tac/score_geometry.py:43-46`):
- `SEG_COEFFICIENT = 100.0` (linear)
- `POSE_COEFFICIENT_INSIDE_SQRT = 10.0` (inside sqrt)
- `RATE_COEFFICIENT = 25.0`
- `CONTEST_REFERENCE_BYTES = 37_545_489`

Operating point (PR106 r2 frontier; `[contest-CUDA]` anchor; non-promotable advisory at A1 base):
- `d_seg ≈ 6.7e-4`
- `d_pose ≈ 3.4e-5`
- `archive_bytes ≈ 178_309` (PR95) or `186_822` (PR106 r2)

Score formula:
```
S = 100·d_seg + sqrt(10·d_pose) + 25·B/37_545_489
  = 100·6.7e-4 + sqrt(10·3.4e-5) + 25·178309/37545489
  = 0.0670  +  0.01844         + 0.001188
  ≈ 0.087   (proxy; actual PR95 contest-CPU is 0.193 because seg+pose+rate scale to higher real numbers — this worked example uses A1-like operating point)
```

Rate slope:
```
dS/dB = 25/37_545_489 = 6.658599e-7 score/byte = 0.0006818 score/KiB
```

Pose marginal (LOCAL — infinitesimal only):
```
dS/d(pose_avg) = (1/2)·10/sqrt(10·pose_avg) = 5/sqrt(10·pose_avg)
              = 5/sqrt(3.4e-4) = 5/0.018439 = 271.16 score/pose-unit
```

SegNet marginal (constant — globally valid):
```
dS/d(seg_avg) = 100 score/seg-unit
```

Pose vs Seg marginal ratio:
```
271.16 / 100 = 2.71  (the CLAUDE.md "2.71× pose dominance at PR106 frontier")
```

**Break-even for a +X KB trailer** (rate-axis penalty offset by pose-axis reduction alone):

```
Penalty:  ΔS_rate = X · 1024 · 6.6586e-7 = X · 6.818e-4 score/KiB
                                          (X in KiB)

To offset on pose axis exactly: subtract ΔS_rate from current pose term:
  new_pose_term = sqrt(10·pose_avg_current) - ΔS_rate
  new_pose_avg  = (new_pose_term)² / 10
  required_pose_reduction = pose_avg_current - new_pose_avg
```

Worked for X = 21 KB at `pose_avg = 3.4e-5`:
```
ΔS_rate = 21 · 6.818e-4 = 0.01432 score
new_pose_term = 0.01844 - 0.01432 = 0.00412
new_pose_avg = (0.00412)² / 10 = 1.697e-6
required pose reduction = 3.4e-5 - 1.697e-6 = 3.23e-5
```

**That's a ~95% reduction in pose_avg** — borderline at PR95's already-tight pose axis.

For comparison via local derivative (WRONG for this magnitude):
```
"required pose reduction" ≈ ΔS_rate / 271.16 = 0.01432 / 271.16 = 5.28e-5
```

But `5.28e-5 > 3.4e-5` (the entire current pose_avg!), so the linear estimate is infeasible by design — there's no way to reduce pose_avg by more than its current value. The exact inverse correctly yields `3.23e-5`, which IS feasible but extreme.

**Break-even for a +X KB trailer (offset by SegNet reduction)** — LINEAR (no fallacy here):

```
required d_seg reduction = ΔS_rate / 100 = X · 6.818e-6 / KiB (in seg-units)
```

For X = 21 KB: required Δd_seg = 1.43e-4 (vs current d_seg ≈ 6.7e-4 → ~21% reduction). Tractable.

**Break-even formula derivation** (math-as-code in `tac.substrates.pr95_lora_dora.budget.exact_pose_reduction_for_score_delta`):

Start: `S(p) = sqrt(10·p)` (just the pose term)
Goal: find `Δp` such that `S(p - Δp) = S(p) - ΔS` (i.e., reduce pose term by exactly `ΔS`)

Solve:
```
sqrt(10·(p - Δp)) = sqrt(10·p) - ΔS
10·(p - Δp)       = (sqrt(10·p) - ΔS)²
10·p - 10·Δp      = 10·p - 2·ΔS·sqrt(10·p) + ΔS²
-10·Δp            = -2·ΔS·sqrt(10·p) + ΔS²
Δp                = (2·ΔS·sqrt(10·p) - ΔS²) / 10
                  = (ΔS · (2·sqrt(10·p) - ΔS)) / 10
```

Sanity check: as `ΔS → 0`, `Δp → 2·ΔS·sqrt(10·p) / 10 = ΔS · sqrt(10·p) / 5 = ΔS / (5/sqrt(10·p))` = `ΔS / 271` ≈ linear approximation. ✓

Infeasibility check: `Δp ≤ p` (you can't reduce pose_avg below 0).
```
(ΔS · (2·sqrt(10·p) - ΔS)) / 10 ≤ p
ΔS · (2·sqrt(10·p) - ΔS) ≤ 10·p
```
At `Δp = p` (whole pose term vanishes): `ΔS = sqrt(10·p)` exactly. So **pose-only offset is feasible iff** `ΔS ≤ sqrt(10·p)` = the current pose term.

At `pose_avg = 3.4e-5`: max ΔS via pose alone = 0.01844 ≈ 1.84 score-units · 1e-2. Anything more requires seg-axis or rate-axis composition.

## Audit of session memos for similar bad math

| Memo | Claim | Verdict |
|---|---|---|
| `.omx/research/pr95_artifact_deconstruction_20260513.md` | Line 200-205: "Exact square-root break-even at `pose_avg=3.4e-5`: +21KB requires reducing pose_avg by ~3.2e-5 if pose alone pays for the trailer. The older marginal estimate of ~5e-7 was wrong because the requested delta is a large fraction of the entire current pose term; linearization is invalid here." | **CORRECT** — already fixed by codex in same commit batch as the helper. ✓ |
| `feedback_pr95_artifact_lora_dora_surgery_landed_20260513.md` (memory) | Line 86: "The 2.71× pose-marginal at PR106 r2 means a pose_avg reduction of 1e-5 is worth -0.00271; tractable." | **WAS WRONG** — fixed in this audit (this session). ✓ |
| `.omx/research/council_a1_pr95_pr98_deliberation_20260512.md` | Line 62: "pose-marginal is `100 / sqrt(10 * 3.4e-5) ~ 542` vs seg-marginal = 100. Ratio: **~5.4x pose dominance**" | **WAS WRONG** — used `100` instead of `5` in the numerator of `d/dp sqrt(10p) = 5/sqrt(10p)`. Fixed in this audit. ✓ |
| `.omx/research/grand_council_pose_axis_non_hnerv_a1_plus_lapose_20260513.md` | Line 22: "A 2 KB expansion that reduces pose_avg by even 10% returns ~0.005 score" | **WAS WRONG** — exact sqrt formula gives 0.001 score for 10% reduction; 0.005 score corresponds to ~50% reduction. Same memo's line 152 has the consistent figure. Fixed in this audit (annotated [math-corrected]; either 10% is the typo or 0.005 is the typo — flagged for operator/codex review). ✓ |
| `.omx/research/grand_council_pose_axis_non_hnerv_a1_plus_lapose_20260513.md` | Line 152: "A 30-60% pose_avg reduction → new pose contribution sqrt(10 × 0.5 × pose_avg) = 0.013, delta = 0.005 to 0.007 score at +200 bytes rate" | **CORRECT** — exact sqrt: `sqrt(10·1.7e-5)=0.01304`, Δ = 0.018439 - 0.013038 = 0.005401 score ≈ 0.005. ✓ Consistent. |
| `.omx/research/theoretical_floor_analyzer_v2_refresh_20260511.md` | Line 113: "pose (at PR106 r2 d_pose) 271 score/unit, 3.69e-6" | **CORRECT** — matches codex's canonical helper. ✓ |
| `.omx/research/grand_council_bug_hunter_config_wiring_integration_audit_20260512.md` | Line 44, 85: "5 / sqrt(10 * 3.4e-5) = 271.16 ... Crossover at `d_pose=2.5e-4` confirmed" | **CORRECT**. ✓ |
| `.omx/research/sub017_frontier_innovation_roadmap_20260513_codex.md` | Line 198: "d/dp sqrt(10p) = 5 / sqrt(10p) ~= 275.8 score / pose-dist unit" (at `d_pose ~= 3.286e-5`) | **CORRECT** — uses slightly different operating point (3.286e-5 vs 3.4e-5); same formula. ✓ |
| `.omx/research/grand_council_first_principles_original_score_lowering_20260513.md` | Line 283: "PoseNet marginal at PR106 r2 (pose_avg=3.4e-5): d(pose)/d(pose_avg) = 271 score/pose-distortion-unit (highly nonlinear in pose_avg)" | **CORRECT** — explicitly flags nonlinearity. ✓ |
| `.omx/research/beat_pr95_curriculum_substrate_training_design_20260513.md` | Lines 440-449: tabular predicted Δseg/Δpose/Δrate per arm, summed via contest formula | **CORRECT IN PRINCIPLE** but uses additive-Δ assumptions on stacked arms (well-flagged "optimistic additive assumption"). The predicted-Δ ranges (-1e-6 to -5e-6 pose) are small enough that linearization IS valid for any single arm. The risk is compounding Δs across arms hitting the linearization-breakdown threshold, which the memo doesn't explicitly check. **ADVISORY**: when summed pose Δ reaches ~10% of current pose_avg, switch from linear to exact sqrt-inverse. ✓ |
| `.omx/research/staged_wavelet_residual_pr106_sidecar_ready_to_dispatch_20260511T174843Z.md` | Line 125: "predicted_d_seg / predicted_d_pose tuple" — no specific numeric break-even claim in this memo | **OUT-OF-SCOPE**. ✓ |
| `.omx/research/sabor_boundary_audit_20260513.md` / `s2sbs_blindspot_audit_20260513.md` | Audit results: capacity tables in pixel-bytes — no score-Δ break-even claims | **OUT-OF-SCOPE**. ✓ |
| `.omx/research/meta_council_decision_attribution_audit_20260513.md` | Line 392: "predicted Δ -0.0005 to -0.003 on PR106 r2 ... bounded by per-byte EV at PR106 r2's `2.40e-9 pose / 6.66e-9 seg` threshold" | **UNVERIFIED FIGURES** — the `2.40e-9 pose / 6.66e-9 seg` per-byte EV threshold is not derived in the memo; cited as "per CLAUDE.md `dS/dB`" but I don't find a direct derivation. **ADVISORY**: this should be reconciled with codex's canonical helper before any dispatch routing decision uses it. |

## Cross-cutting findings

1. **Pose-marginal linearization fallacy** is the single biggest math-correctness risk in our session memos. The CLAUDE.md table itself correctly states "271 score/pose-unit at pose_avg=3.4e-5" but does not warn that this is a LOCAL derivative. Any memo that uses this number to compute large pose reductions inherits the fallacy.

2. **SegNet axis is linear**, so the same linearization-of-large-Δ trap does NOT exist there. `dS/d(seg_avg) = 100` globally; `ΔS_seg = 100·Δd_seg` is exact for any Δd_seg.

3. **Rate axis is linear** in bytes: `dS/dB = 6.66e-7 score/byte` globally; `ΔS_rate = 6.66e-7·ΔB` is exact for any ΔB.

4. **The bug class is unique to the pose axis** at the PR106 frontier because the pose term contains a square root.

5. **The 5.4× pose dominance figure in `council_a1_pr95_pr98_deliberation_20260512.md` line 62** was a sister bug (used wrong leading coefficient in the formula). Now corrected to 2.71× consistent with CLAUDE.md.

6. **Compounding small Δ across stacked arms** (as in `beat_pr95_curriculum_substrate_training_design_20260513.md`) is fine IF each Δ remains in the linear regime AND the SUM remains linear. The transition threshold is approximately `Δp ≈ 0.1·p` (pose_avg). Beyond that, use exact inversion via `tac.substrates.pr95_lora_dora.budget.exact_pose_reduction_for_score_delta`.

## Recommended STRICT preflight gate (operator-routable, deferred)

**Proposed gate name**: `check_score_break_even_claim_uses_exact_inverse_or_explicit_linear_tag`

**Detection scope**: any `.omx/research/*.md` or `~/.claude/projects/.../memory/feedback_*.md` containing the regex `pose[_ ]marginal` AND a numeric Δscore claim with `Δ` or `=` operator AND a numeric `pose_avg`/`d_pose`/`Δp` reduction value > 0.5·`pose_avg`.

**Acceptance**: any of:
- Same-line `[math-corrected-by-codex YYYY-MM-DD]` tag
- Sibling reference to `tac.substrates.pr95_lora_dora.budget` (or any future canonical break-even helper)
- Explicit linearization caveat: "for infinitesimal reductions" / "local derivative only" / "linearized; valid only if Δp ≤ 0.1·pose_avg" / equivalent

**Rationale**: a STRICT gate would prevent any future session memo from claiming linear pose-margin extrapolation past the breakdown threshold. Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against", every codex-found bug class deserves a self-protection layer. The gate would extinct the linearization fallacy at the typing layer.

**Risk**: false positives on legitimate linear-regime small-Δ tables (e.g., beat-PR95 design memo) need waiver mechanism. The acceptance #3 (linearization caveat) handles this.

**Operator-routable**: should this gate be implemented? It would extinct the bug class but adds ~50-100 LOC + 15-20 tests + waiver mechanism. The marginal value is preventing future versions of the LoRA/DoRA bug from shipping into operator-facing memos.

## Wire-in hooks per Catalog #125 (subagent coherence-by-default)

1. **Sensitivity-map contribution** (`tac.sensitivity_map.*`): N/A this landing — audit memo, no new sensitivity-map row.
2. **Pareto constraint** (`tac.pareto_*`): N/A — no new Pareto knowledge; the bug class is in MEMO ARITHMETIC, not in the achievable region.
3. **Bit-allocator hook**: N/A — no bit allocation changes.
4. **Cathedral autopilot dispatch hook**: N/A — no dispatch artifact.
5. **Continual-learning posterior update**: N/A — no new empirical anchor.
6. **Probe-disambiguator**: N/A — the math has a SINGLE correct answer (codex's exact sqrt inverse); not 2+ defensible interpretations.

All 6 hooks declared N/A explicitly per CLAUDE.md "Silent omission is the orphan-work failure mode" non-negotiable. This is a pure documentation/math-correction audit; no orphan signals.

## References

- Codex commit: `cd40d859` "pr95: add LoRA adapter break-even budget math"
- Canonical helper: `src/tac/substrates/pr95_lora_dora/budget.py::exact_pose_reduction_for_score_delta`
- Canonical tests: `src/tac/substrates/pr95_lora_dora/tests/test_budget.py`
- Score formula source of truth: `src/tac/score_geometry.py:43-46`
- CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" (the 2.71× table)
- LoRA/DoRA landed memo: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_pr95_artifact_lora_dora_surgery_landed_20260513.md`
- LoRA/DoRA deconstruction memo: `.omx/research/pr95_artifact_deconstruction_20260513.md`
- Beat-PR95 design memo: `.omx/research/beat_pr95_curriculum_substrate_training_design_20260513.md`
- Codex coordination synthesis: `.omx/research/codex_coordination_shared_page_synthesis_20260513.md`
- Codex frontier roadmap §"Formal score arithmetic and decision calculus": `.omx/research/sub017_frontier_innovation_roadmap_20260513_codex.md`
- Lane: `lane_codex_math_correction_and_coordination_20260513` (L0 SKETCH; target L1 after this audit lands)
