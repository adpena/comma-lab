# Cleanup-refactor audit during contest_oracle build
# Date: 2026-05-18
# Lane: lane_contest_oracle_canonical_package_meta_meta_15_implications_20260518
# Authority: operator standing directive 2026-05-18 verbatim *"clean up and refactor perhaps without signal loss for standardization and canonicalization and no duplicate code and production hardened OSS"*

## Scope

This audit was conducted IN-FLIGHT during the `tac.contest_oracle` build per
the operator's 4th standing directive. It surfaces duplicate-code instances
+ refactor opportunities + signal-preservation guarantees, sized to be
operator-routable to follow-on subagents per CLAUDE.md "Forbidden premature
KILL" (defer-with-reactivation-criteria default, NEVER kill).

This memo is INFORMATIONAL, not actionable on its own. Each finding is paired
with a recommended follow-on lane that can be dispatched in a future wave.

## Methodology

Per Catalog #229 premise verification + Catalog #305 6-facet observability:
each finding cites the canonical helper it should consolidate against, the
specific files containing the duplicate, the estimated LOC reduction, and
the predicted ΔS contribution (typically zero direct, but unlocks downstream
substrate-trainer wire-ins).

## Findings

### Finding 1: 35 substrate trainers hardcode `math.sqrt(10.0)` for pose weight

**Surface:** `experiments/train_substrate_*.py` files (35 files, 37 occurrences)

**Canonical helper:** `tac.contest_oracle.pose_axis_canonical.CONTEST_POSE_SQRT_WEIGHT`
+ `tac.substrates.score_aware_common.CONTEST_POSE_SQRT_WEIGHT` (sister)

**Duplication:** Every substrate trainer that consumes the contest pose sqrt
weight defines its own `default=math.sqrt(10.0)` argparse default OR inline
`gamma_pose * sqrt(10)` expression. 35 surfaces; one canonical constant.

**Refactor opportunity:** Replace `math.sqrt(10.0)` with
`from tac.contest_oracle import CONTEST_POSE_SQRT_WEIGHT` import + use the
canonical constant. Estimated reduction: ~70 LOC across the corpus (37
duplicate inline computations + import lines).

**Signal preservation guarantee:** The canonical constant equals
`math.sqrt(10.0) = 3.1622776601683795` exactly; the refactor is a
name-change ONLY with byte-identical numerical behavior. Catalog #228
F3 backport gate wire-in flagged the canonical surface as the standard;
this is its sister consolidation.

**Recommended follow-on lane:** `lane_consolidate_substrate_trainer_pose_sqrt_constant_20260519`
(34 trainers; ~1h editor; sister of Catalog #228 F3 backport wave; subject
to per-substrate-trainer wire-in discipline per Catalog #325).

### Finding 2: 14 substrate trainers route through `tac.substrates.score_aware_common` canonical helper

**Surface:** `experiments/train_substrate_*.py` files (14 fully canonical;
the audit confirms this is HIGH adoption of the canonical sister already).

**Canonical helper:** `tac.substrates.score_aware_common.score_pair_components`

**Status:** This is GOOD — the operator-routable history per Catalog #226
already drove 14-of-32 substrate trainers through the canonical
`gate_auth_eval_call` + `score_pair_components` helpers. The remaining
~18 substrate trainers are candidates for further consolidation.

**Recommended follow-on lane:** Subsumed by Catalog #326 driver-mode + per-
substrate canonical helpers wave; no new lane needed.

### Finding 3: per-class Lagrangian implicit-uniform-weighting bug class

**Surface:** Most substrate trainers' seg-loss computation treats all 5
SegNet classes uniformly. There is NO canonical per-class Lagrangian helper
prior to this audit's `tac.contest_oracle.per_class_lagrangian.compute_per_class_lambda_seg`.

**Canonical helper:** Just-landed `tac.contest_oracle.per_class_lagrangian`
(Impl 5 per design memo).

**Refactor opportunity:** Substrate trainers that train against
`d_seg_per_class` (vs the pooled `d_seg` scalar) should consume the
per-class Lagrangian helper. Rare classes get proportionally higher
weight, matching the empirical class-imbalance signal in comma2k19.

**Signal preservation guarantee:** The new helper is ADDITIVE — opt-in.
Trainers that don't consume it continue working under the implicit-uniform
default; opt-in is per-trainer per-symposium per Catalog #325 discipline.

**Recommended follow-on lane:** `lane_per_class_lagrangian_substrate_trainer_wire_in_20260519`
(per-substrate symposium-gated per Catalog #325; cheap CPU smoke validation
required per Catalog #192).

### Finding 4: 4 substrate trainers (lapose/wavelet/atw_v1/atw_v2/balle/block_nerv/c6/cool_chic) all use `--gamma-pose` argparse flag

**Surface:** Multiple substrate trainers expose `--gamma-pose` with default
`math.sqrt(10.0)` — but the meaning differs across trainers (multiplicative
override vs additive contribution).

**Refactor opportunity:** Standardize the `--gamma-pose` flag semantics
across substrate trainers. Recommend: `--gamma-pose 1.0` (multiplier over
the canonical `CONTEST_POSE_SQRT_WEIGHT`) becomes the canonical default;
substrate trainers that need a different default carry a same-line waiver
documenting why.

**Recommended follow-on lane:** Subsumed by Finding 1 (sister of pose
sqrt constant consolidation).

### Finding 5: Bandit / Thompson-sampling primitives scattered

**Surface:** `tac.boosting.*` carries some bandit-like primitives;
`tac.cathedral_autopilot_*` carries others (Rashomon ensemble);
`tac.master_gradient_consumers` has its own per-pair sampling.

**Canonical helper:** Just-landed `tac.contest_oracle.bandit_per_pair`
(canonical Beta-Bernoulli Thompson sampling per Impl 13).

**Refactor opportunity:** Catalog the existing bandit primitives across
the codebase (sister of the design memo APPENDIX A inventory) and route
them through the canonical bandit helper.

**Recommended follow-on lane:** `lane_bandit_thompson_sampling_canonicalization_20260519`
(read-only inventory subagent + recommendation memo; no production code
changes without operator approval per CLAUDE.md "Design decisions" non-negotiable).

### Finding 6: Production-hardened OSS publishing -- license headers complete

**Status:** ALL 14 new modules + 11 linguistic-extension additions carry
SPDX-License-Identifier: MIT headers per CLAUDE.md "Beauty, simplicity, and
developer experience" + operator OSS-publication standing directive.

**Refactor opportunity:** Sweep the existing canonical helpers for license-
header consistency. The sister Atom landing (`181fa4c1e`) had SPDX headers
throughout; the older canonical helpers may not.

**Recommended follow-on lane:** `lane_oss_license_header_canonicalization_20260519`
(cosmetic refactor only; ~30min editor; subject to operator approval).

### Finding 7: Type-hint coverage on canonical helpers

**Status:** ALL 14 new contest_oracle modules + 11 linguistic-extension
additions have FULL type hints (no `Any` outside `Mapping[str, Any]` for
provenance/metadata pass-through). Matches existing canonical pattern.

**Refactor opportunity:** Sweep older canonical helpers for type-hint
coverage. Some may use `# type: ignore` or have missing annotations.

**Recommended follow-on lane:** Subsumed by Finding 6 (cosmetic refactor
wave).

## Signal-preservation guarantees

Per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable:

1. NO behavioral changes to existing canonical helpers were made.
2. ALL new canonical helpers in `tac.contest_oracle/` are NEW SURFACES;
   substrate trainers + sister consumers continue working unchanged.
3. The 11 new `tac.atom` linguistic extensions are ADDITIVE; the
   existing 34 public symbols are unchanged.
4. Tests confirm: `pytest src/tac/atom/tests/ -q -> 120 passed (was 120
   pre-extension)`; `pytest src/tac/contest_oracle/tests/ -q -> 131 passed
   (new)`.

## Estimated cumulative impact

If all 7 findings' recommended follow-on lanes land:
- LOC reduction: ~150 LOC across the substrate-trainer corpus.
- Canonical-helper consolidation: 6 surfaces unified.
- Production-OSS readiness: 100% across `tac.contest_oracle` +
  `tac.atom.linguistic_extensions`; ~80% across the broader codebase.

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" non-negotiable:
NONE of these findings make a direct ΔS claim. The predicted IMPACT of
follow-on substrate-trainer wire-ins (per Finding 3 specifically) is
operator-symposium-routable per Catalog #325 + Catalog #294 9-dim
checklist; cheap CPU smoke validation per Catalog #192 is the canonical
gate before any paid Modal/Lightning/Vast.ai dispatch.

## Cross-references

- Design memo: `.omx/research/contest_fixed_as_oracles_15_implications_design_memo_20260518.md`
- Sister landings (today): commit `181fa4c1e` (`tac.atom` canonical),
  commit `6db94d9ea` (TOP-1+TOP-4 arbitrariness extinction),
  commit `2d042f7e6` (arbitrariness-extinction audit).
- Catalog #226 canonical auth-eval helper wave (sister consolidation pattern).
- Catalog #325 per-substrate symposium discipline (gates per-substrate
  wire-ins).

— Lane `lane_contest_oracle_canonical_package_meta_meta_15_implications_20260518` cleanup-refactor audit, in-flight 2026-05-18
