# SPDX-License-Identifier: MIT
"""Canonical rename wave D7+D8+D9 — 3 NEW canonical anti-patterns.

Per operator-approved 2026-05-30 "All are approved, land inline" + canonical
mapping memo at `.omx/research/canonical_rename_wave_mapping_20260530.md`
sections D7+D8+D9. Sister of `tac.canonical_anti_patterns.builtins` at the
canonical-rename-wave sub-surface; landed in a separate module per
UNIQUE-AND-COMPLETE-PER-METHOD operating mode + Catalog #290 (each
substantive anti-pattern wave gets its own focused builder file so future
readers see the canonical-rename context at a glance).

The 3 new anti-patterns close 3 orthogonal local-minima-perpetuation
failure modes that drove the 0.196-0.199 plateau across 8 months of
operator effort (per `[[feedback-assumptions-challenge-audit-break-out-local-minima]]`
18-shared-assumption empirical audit + commit `6d3c42635` canonical rename
wave D1+D2):

D7: `canonical_default_plateau_substrate_disguised_as_class_shift_v1`
    Substrate-engineering paradigm. HIGH severity. 30+ substrates we
    "built" were HNeRV variants in disguise; each adopted the canonical
    18-assumption profile by default rather than explicitly forking
    assumptions to achieve genuine class-shift.

D8: `micro_optimization_without_macro_escape_polishing_plateau_ceiling_v1`
    Optimization-budget paradigm. HIGH severity. Burning compute on
    within-class refinements that cannot break the plateau ceiling.

D9: `hnerv_pr95_language_anchoring_local_minima_perpetuation_v1`
    Documentation-discipline paradigm. MEDIUM severity. Language anchoring
    to architecture-class label keeps subagents producing variations on
    the same thing even when the operator's intent is class-shift; the
    canonical rename wave D1+D2 at commit `6d3c42635` IS the canonical
    extinction event (HARDENED-EARNED EmpiricalFalsification: documented).

Cross-references:
  * `tac.canonical_anti_patterns.builtins` — sister 17 initial anti-patterns
  * `tac.canonical_anti_patterns.registry.register_anti_pattern` — canonical
    persistence helper this module routes through
  * Catalog #344 — canonical anti-patterns + canonical equations registry
  * Catalog #287 — placeholder-rationale rejection sister discipline
  * Catalog #309 — horizon_class taxonomy (D7 routes plateau_adjacent_substrate)
  * Catalog #290 — UNIQUE-AND-COMPLETE-PER-METHOD design-memo discipline
  * Catalog #325 — per-substrate optimal-form symposium 6-step contract
  * Catalog #296 — Dykstra-feasibility predicted-band check
  * Catalog #110/#113 — APPEND-ONLY HISTORICAL_PROVENANCE
  * Commit `6d3c42635` — D1+D2 canonical rename wave inline landing (D9 anchor)
  * `.omx/research/canonical_rename_wave_mapping_20260530.md` — full D1-D12 plan
"""
from __future__ import annotations

from pathlib import Path

from tac.canonical_anti_patterns.anti_pattern import (
    INCIDENT_IMPLEMENTATION_LEVEL_CONFIRMATION,
    INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE,
    PARADIGM_DISCIPLINE,
    PARADIGM_RIGOR_LOSS,
    RECALIBRATE_ON_NEW_FALSIFICATIONS,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_OBSERVED_HIGH,
    SEVERITY_OBSERVED_MEDIUM,
    AntiPattern,
    EmpiricalFalsification,
)
from tac.canonical_anti_patterns.registry import register_anti_pattern
from tac.provenance.builders import build_provenance_for_predicted
from tac.provenance.contract import Provenance


# Canonical landing UTC for D7+D8+D9 registration wave.
_D7_D8_D9_LANDING_UTC = "2026-05-30T22:00:00Z"

# Canonical placeholder SHA for design-only provenance (mirrors builtins).
_DESIGN_PROV_SHA = "0" * 64

# Canonical commit anchor for D9 EmpiricalFalsification.
# Commit 6d3c42635 = D1+D2 canonical rename wave inline landing 2026-05-30
# (operator-approved "All are approved, land inline" per canonical mapping memo).
_D1_D2_RENAME_COMMIT_SHA = "6d3c42635"


def _design_provenance(anti_pattern_id: str) -> Provenance:
    """Build a PREDICTED Provenance for canonical-rename-wave anti-pattern.

    Mirrors `tac.canonical_anti_patterns.builtins._design_provenance`.
    Anti-patterns are CLASS-level predictions of future recurrences; they
    are never promotable score claims. The PREDICTED grade + non-promotable
    invariants are enforced at construction by
    `build_provenance_for_predicted` per Catalog #323.
    """
    return build_provenance_for_predicted(
        model_id=(
            f"canonical_anti_patterns.d7_d8_d9_builders.{anti_pattern_id}"
        ),
        inputs_sha256=_DESIGN_PROV_SHA,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
        captured_at_utc=_D7_D8_D9_LANDING_UTC,
    )


# -----------------------------------------------------------------------------
# D7 — canonical_default_plateau_substrate_disguised_as_class_shift_v1
# -----------------------------------------------------------------------------
#
# Operator binding directive 2026-05-30 verbatim: "any language like hnerv.or
# pr 95 parity language that might keep us stuck in local minima despite good
# intentions". The HNeRV variants in disguise pattern is the canonical
# substrate-engineering failure mode behind the 0.196-0.199 plateau cluster.
#
# Empirical anchors enumerated from the 18-shared-assumption empirical audit
# (`feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md`):
# the audit documented 30+ substrates sharing 18 structural assumptions (EMA
# 100% / archive.zip 100% / eval_roundtrip 97% / canonical scorer-preprocess
# 97% / canonical auth_eval routing 97% / Tier-1 engineering 78-100%). Each
# of these substrates was claimed as a class-shift candidate but adopted the
# canonical default 18-assumption profile rather than explicitly forking.
#
# HONESTY DISCIPLINE per CLAUDE.md NO FAKE IMPLEMENTATIONS non-negotiable:
# we register 1 substantive EmpiricalFalsification anchoring the audit's
# canonical findings (the 18-assumption matrix is queryable; the audit IS
# the canonical empirical landing memo for this anti-pattern). Enumerating
# 30+ individual substrate confirmations as separate falsifications would
# be ceremony without signal — the canonical audit memo aggregates them.
# Future per-substrate audits can append additional falsifications via the
# canonical `append_empirical_falsification` helper.


def build_canonical_default_plateau_substrate_disguised_as_class_shift_v1() -> AntiPattern:
    """Anti-pattern D7: HNeRV-variant-in-disguise substrate adopting 18-default assumptions.

    Empirical anchor: 18-shared-assumption audit 2026-05-15 documented
    30+ substrates claimed as class-shift but structurally HNeRV variants
    via canonical-default-adoption of the 18 shared assumptions (SA01-SA18
    in audit's matrix table). Each substrate adopted EMA decay 0.997 +
    archive.zip monolithic + eval_roundtrip + canonical scorer-preprocess
    + Tier-1 engineering uniformly — and produced scores clustered in the
    0.196-0.199 plateau because the 10% variance between substrates was
    the only variance NOT shared.
    """
    return AntiPattern(
        anti_pattern_id=(
            "canonical_default_plateau_substrate_disguised_as_class_shift_v1"
        ),
        description=(
            "Substrate claimed as class-shift candidate adopts the canonical "
            "default 18-assumption profile per `[[feedback-assumptions-challenge-"
            "audit-break-out-local-minima]]` audit (EMA 100% / archive.zip "
            "100% / eval_roundtrip 97% / canonical scorer-preprocess 97% / "
            "Tier-1 engineering 78-100%) without explicit per-assumption "
            "FORK rationale. Result: substrate is structurally a HNeRV "
            "variant in disguise; scores cluster in the 0.196-0.199 plateau "
            "because the 10% variance between substrates is the only "
            "variance NOT shared."
        ),
        forbidden_pattern_predicate=(
            "substrate_design_memo declares class_shift_substrate OR "
            "lane_class=frontier_pursuit AND fails 18-assumption profile "
            "audit (>=80% of SA01-SA18 ADOPTED rather than FORKED with "
            "substantive rationale)"
        ),
        falsification_band={
            # The plateau cluster the canonical-default-adoption pattern
            # empirically produces (0.196-0.199). Score values are honest
            # empirical anchors from the assumption-audit memo PVs.
            "plateau_cluster_score_lo": 0.196,
            "plateau_cluster_score_hi": 0.199,
            # The shared-assumption-prevalence threshold above which a
            # substrate is structurally a class-shift candidate in name
            # but a within-class refinement in implementation.
            "shared_assumption_prevalence_threshold": 0.80,
        },
        recurrence_conditions=(
            "substrate trainer adopts EMA decay 0.997 without per-substrate "
            "empirical justification",
            "substrate adopts canonical scorer-preprocess routing without "
            "per-substrate gradient-path audit",
            "substrate adopts archive.zip monolithic grammar without "
            "per-substrate grammar-shift evaluation",
            "substrate adopts eval_roundtrip=True uniformly without "
            "per-substrate quantization-roundtrip-bandwidth audit",
            "substrate adopts canonical Tier-1 engineering primitives "
            "uniformly without per-substrate cost-band reconciliation",
            "substrate design memo lacks `## 18-shared-assumption profile "
            "registration` section per D4 + D5 canonical mapping",
            "substrate design memo lacks `## Canonical-vs-unique decision "
            "per layer` section per Catalog #290",
            "substrate fails per-substrate symposium PROCEED-unconditional "
            "verdict per Catalog #325 before paid dispatch",
        ),
        canonical_source_anchor=(
            "operator binding META-correction 2026-05-30 verbatim 'any "
            "language like hnerv.or pr 95 parity language that might keep "
            "us stuck in local minima despite good intentions'; "
            "feedback_assumptions_challenge_audit_break_out_local_minima_"
            "landed_20260515.md 18-assumption empirical audit; "
            ".omx/research/canonical_rename_wave_mapping_20260530.md D7 "
            "canonical anti-pattern specification; commit 6d3c42635 D1+D2 "
            "canonical rename wave inline landing"
        ),
        canonical_unwind_path=(
            "Audit per-substrate 18-assumption profile per D4 + D5; classify "
            "each assumption as ADOPTED-vs-FORKED with substantive rationale; "
            "if >=80% ADOPTED (= within-class) declare lane_class="
            "plateau_adjacent_substrate per Catalog #309 horizon class "
            "taxonomy; require Catalog #325 per-substrate symposium PROCEED-"
            "unconditional before paid dispatch; per Catalog #290 the design "
            "memo MUST include `## 18-shared-assumption profile registration` "
            "section enumerating per-assumption ADOPT/FORK decision; per "
            "Catalog #296 the predicted-ΔS band MUST cite Dykstra-feasibility "
            "intersection check against the existing canonical frontier."
        ),
        canonical_producers=(
            "experiments/train_substrate_*.py",
            ".omx/research/*_design_*.md",
            ".omx/state/lane_registry.json",
        ),
        canonical_consumers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
            "src/tac/preflight.py (Catalog #290/#294/#296/#303/#305/#325)",
            "src/tac/canonical_anti_patterns.match_stack_against_anti_patterns",
        ),
        paradigm_class=PARADIGM_DISCIPLINE,
        severity=SEVERITY_HIGH,
        provenance=_design_provenance(
            "canonical_default_plateau_substrate_disguised_as_class_shift_v1"
        ),
        empirical_falsifications=(
            EmpiricalFalsification(
                anti_pattern_id=(
                    "canonical_default_plateau_substrate_disguised_as_class_shift_v1"
                ),
                falsification_id=(
                    "assumptions_challenge_audit_18_shared_assumption_matrix_20260515"
                ),
                measurement_method=(
                    "grep-based empirical prevalence audit across 32 substrate "
                    "trainers under experiments/train_substrate_*.py per audit "
                    "memo PV6+PV10; matrix table SA01-SA18 prevalence quantified "
                    "100% (eval contract) / 97% (canonical helpers) / 78-100% "
                    "(Tier-1 engineering)"
                ),
                empirical_artifact_path=(
                    "memory:feedback_assumptions_challenge_audit_break_out_"
                    "local_minima_landed_20260515.md"
                ),
                empirical_output={
                    "assumptions_audited": 18,
                    "shared_assumption_prevalence_max": 1.00,
                    "shared_assumption_prevalence_min": 0.78,
                    "shared_assumption_prevalence_mean_approx": 0.94,
                    "substrates_audited": 32,
                    "plateau_cluster_score_lo_observed": 0.19285,
                    "plateau_cluster_score_hi_observed": 0.19870,
                    "audit_pvs_documented": 11,
                    "canonical_consumer_score_pair_components_prevalence": 0.97,
                    "ema_decay_0_997_uniform_prevalence": 1.00,
                },
                falsification_residual=0.0,
                captured_at_utc="2026-05-15T00:00:00Z",
                canonical_provenance=_design_provenance(
                    "canonical_default_plateau_substrate_disguised_as_class_shift_v1"
                ),
                incident_classification=(
                    INCIDENT_IMPLEMENTATION_LEVEL_CONFIRMATION
                ),
                severity_observed=SEVERITY_OBSERVED_HIGH,
                operator_routable_unwind_path=(
                    "Per audit Top-10 substrate-class-shift NSCS proposals + "
                    "D4 + D5 18-assumption registration discipline: every NEW "
                    "substrate design memo MUST include per-assumption ADOPT/"
                    "FORK decision before paid dispatch; cathedral autopilot "
                    "ranker consumes the canonical anti-pattern to discount "
                    "plateau-adjacent substrates per Catalog #309"
                ),
            ),
        ),
        last_recalibration_utc=_D7_D8_D9_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


# -----------------------------------------------------------------------------
# D8 — micro_optimization_without_macro_escape_polishing_plateau_ceiling_v1
# -----------------------------------------------------------------------------
#
# Operator concern (canonical mapping memo D8): "WHAT NOT TO DO" + "the
# importance of synergy and binding and full stack and also micro
# optimization, bridged". Micro-optimization without a macro-escape path
# in flight merely polishes the plateau ceiling and burns compute on
# within-class refinements that cannot break the 0.196-0.199 plateau.
#
# HONESTY DISCIPLINE: this anti-pattern is design-time PREVENTION rather
# than empirically-anchored falsification; the canonical empirical evidence
# is the same 0.196-0.199 plateau cluster D7 codifies. We register the
# anti-pattern with empirical_falsifications=() (design-only at landing)
# and the FIRST falsification will land when a future audit identifies a
# specific session where micro-optimization budget was burned on a
# plateau-adjacent substrate WITHOUT a macro-escape path in flight.


def build_micro_optimization_without_macro_escape_polishing_plateau_ceiling_v1() -> AntiPattern:
    """Anti-pattern D8: micro-optimization budget burned without macro-escape path.

    Within-class refinement (per-tensor mixed-precision / brotli quality
    sweeps / Hessian-saliency reweighting / Tier-1 engineering polishing
    / etc.) on a substrate ABOVE the canonical frontier is plateau
    polishing; the predicted ΔS contribution cannot escape the within-class
    ceiling regardless of budget. Per audit memo Top-10 substrate-class
    shifts: macro-escape paths (NSCS02 downsampled / NSCS01 nullspace
    split / NSCS03 end-to-end joint codec / NSCS04 Wyner-Ziv world model
    / NSCS10 pure Daubechies) deliver -0.005 to -0.080 vs plateau; micro-
    optimization on existing substrates delivers -0.0005 to -0.005.
    """
    return AntiPattern(
        anti_pattern_id=(
            "micro_optimization_without_macro_escape_polishing_plateau_ceiling_v1"
        ),
        description=(
            "Within-class refinement budget allocated to substrate ABOVE "
            "canonical frontier WITHOUT a macro-escape (class-shift) path "
            "in flight or queued. Result: budget burned on plateau polishing; "
            "EV is bounded by within-class ceiling; the binding-depth gap "
            "per `[[pr-or-greater-parity-synergy-binding-integration-not-"
            "hnerv-specific-meta-class-lesson-correction]]` remains intact. "
            "Per audit Top-10: macro-escape delivers -0.005 to -0.080; "
            "within-class refinement delivers -0.0005 to -0.005."
        ),
        forbidden_pattern_predicate=(
            "operator_authorize_recipe declares dispatch on substrate whose "
            "predicted_band per Catalog #296 lies AT-OR-ABOVE canonical "
            "frontier AND no class-shift candidate per Catalog #309 horizon_"
            "class=frontier_pursuit OR asymptotic_pursuit is in flight or "
            "queued in lane registry"
        ),
        falsification_band={
            # Empirical ceiling on micro-optimization EV (from audit Top-10
            # EV ranking; within-class refinement upper bound). Score-deltas
            # are honest empirical anchors from audit memo.
            "micro_optimization_score_delta_ceiling_magnitude": 0.005,
            # Empirical floor on macro-escape EV (from audit Top-10 EV
            # ranking; class-shift lower bound for best candidates).
            "macro_escape_score_delta_floor_magnitude": 0.005,
            # Empirical EV ratio threshold (macro/micro) above which the
            # opportunity cost of micro-optimization dominates.
            "macro_vs_micro_ev_ratio_threshold": 10.0,
        },
        recurrence_conditions=(
            "dispatch wrapper fires substrate refinement on plateau-adjacent "
            "substrate without macro-escape path in lane registry",
            "operator-authorize recipe lacks horizon_class declaration per "
            "Catalog #309",
            "cathedral autopilot ranker top-3 candidates are ALL plateau-"
            "adjacent (no class-shift candidate in queue)",
            "design memo lacks Dykstra-feasibility predicted-ΔS check per "
            "Catalog #296",
            "session burns >$10 paid GPU on within-class refinement WITHOUT "
            "a parallel macro-escape candidate in queue",
            "design memo proposes within-class refinement on substrate "
            "whose latest contest-CUDA or contest-CPU anchor matches the "
            "canonical frontier within 1e-3",
            "session ranks micro-optimization above class-shift per "
            "HIGHEST-EV-SHORTEST-WALL-CLOCK trichotomy per "
            "[[canonical-ev-metric-trichotomy]] despite class-shift "
            "candidate availability",
        ),
        canonical_source_anchor=(
            "operator concern canonical mapping memo D8 'WHAT NOT TO DO' + "
            "'the importance of synergy and binding and full stack and also "
            "micro optimization, bridged'; "
            "feedback_assumptions_challenge_audit_break_out_local_minima_"
            "landed_20260515.md Top-10 substrate-class-shift EV ranking; "
            ".omx/research/canonical_rename_wave_mapping_20260530.md D8 "
            "canonical anti-pattern specification"
        ),
        canonical_unwind_path=(
            "Before allocating budget to micro-optimization (within-class "
            "refinement of existing substrate), verify (a) macro-escape "
            "path is in flight or queued in lane registry per Catalog #309 "
            "horizon_class=frontier_pursuit OR asymptotic_pursuit AND "
            "(b) substrate's predicted ΔS band per Catalog #296 Dykstra-"
            "feasibility check is at-or-below current frontier — micro-"
            "optimization on a substrate ABOVE frontier is plateau "
            "polishing; per Catalog #325 per-substrate symposium MUST score "
            "macro-vs-micro EV ratio explicitly per "
            "[[canonical-ev-metric-trichotomy]] HIGHEST-EV-SHORTEST-WALL-"
            "CLOCK ranking; per Catalog #299 quota brake spirit: prefer "
            "ONE macro-escape dispatch over N micro-optimization dispatches "
            "when macro/micro EV ratio exceeds 10."
        ),
        canonical_producers=(
            ".omx/operator_authorize_recipes/*.yaml",
            ".omx/state/lane_registry.json",
            ".omx/research/*_design_*.md",
            "experiments/train_substrate_*.py",
        ),
        canonical_consumers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
            "src/tac/preflight.py (Catalog #296/#309/#325)",
            "src/tac/canonical_anti_patterns.match_stack_against_anti_patterns",
        ),
        paradigm_class=PARADIGM_DISCIPLINE,
        severity=SEVERITY_HIGH,
        provenance=_design_provenance(
            "micro_optimization_without_macro_escape_polishing_plateau_ceiling_v1"
        ),
        empirical_falsifications=(),
        last_recalibration_utc=_D7_D8_D9_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


# -----------------------------------------------------------------------------
# D9 — hnerv_pr95_language_anchoring_local_minima_perpetuation_v1
# -----------------------------------------------------------------------------
#
# Operator binding META-concern 2026-05-30 verbatim: "any language like
# hnerv.or pr 95 parity language that might keep us stuck in local minima
# despite good intentions". The canonical EXTINCTION EVENT is the D1+D2
# inline canonical rename wave at commit 6d3c42635 2026-05-30 — section
# header line 186 + L14-L32 subsection header line 214 renamed from
# "HNeRV parity discipline" / "PR95-family" to "canonical leaderboard
# binding-depth discipline" / "canonical leaderboard-winning techniques
# validated across PR95/100/101/102/103/106/110".
#
# Per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: HISTORICAL
# references throughout CLAUDE.md catalog rows + canonical equation prefix
# `pr95_family_l<N>_*_v1` are PRESERVED (not mutated). The anti-pattern
# refuses FORWARD-LOOKING text adoption of the local-minima-anchoring
# language; sister Catalog #309 horizon_class taxonomy provides the
# canonical replacement framing.
#
# This is the CANONICAL EmpiricalFalsification anchor: today's D1+D2
# canonical rename wave commit `6d3c42635` IS the empirical extinction
# event. The structural-protection-surfaces measure quantifies how the
# wave hardens forward-looking text without mutating HISTORICAL_PROVENANCE.


def build_hnerv_pr95_language_anchoring_local_minima_perpetuation_v1() -> AntiPattern:
    """Anti-pattern D9: HNeRV/PR95 architecture-class language anchoring.

    Operator caught the META-class pattern: forward-looking text framing
    binding-depth lessons as 'HNeRV parity' or 'PR95-family' anchors
    cognitive frame to a specific architecture class, perpetuating the
    18-shared-assumption local minima even when the operator's intent is
    class-shift. The D1+D2 inline canonical rename wave at commit
    `6d3c42635` IS the canonical extinction event.

    Per CLAUDE.md "Forbidden premature KILL" + Catalog #110/#113 APPEND-
    ONLY HISTORICAL_PROVENANCE: this anti-pattern does NOT kill the
    historical references; it documents the language-anchoring failure
    mode + routes forward-looking text to the canonical replacement
    framing (canonical leaderboard binding-depth discipline / PR-or-
    greater parity / binding-depth discipline per Catalog #309).
    """
    return AntiPattern(
        anti_pattern_id=(
            "hnerv_pr95_language_anchoring_local_minima_perpetuation_v1"
        ),
        description=(
            "Forward-looking text adopts 'HNeRV parity' / 'PR95-family' "
            "architecture-class label rather than canonical binding-depth "
            "framing. Result: subagents reading the text inherit the "
            "architecture-class cognitive frame and produce HNeRV variants "
            "in disguise rather than class-shift candidates. The language "
            "itself doesn't cause the bug — it just doesn't extinct the "
            "underlying 18-shared-assumption anchoring failure mode. The "
            "canonical rename wave 2026-05-30 commit 6d3c42635 IS the "
            "extinction event; HISTORICAL references preserved per "
            "Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE."
        ),
        forbidden_pattern_predicate=(
            "forward-looking text (NEW design memo / NEW landing memo / "
            "NEW canonical equation registration / NEW canonical anti-"
            "pattern registration / NEW catalog row) adopts 'HNeRV parity' "
            "OR 'PR95-family' OR 'HNeRV-family' OR 'PR-95-family' AS PRIMARY "
            "FRAMING for binding-depth discipline rather than canonical "
            "binding-depth language"
        ),
        falsification_band={
            # The canonical rename wave touched 2 of 2 in-scope forward-
            # looking section headers (D1 + D2); HISTORICAL_PROVENANCE
            # references preserved per Catalog #110/#113. This anchors
            # the canonical "structural protection at 2 surfaces, zero
            # mutation of historical" extinction signature.
            "forward_looking_section_headers_renamed": 2.0,
            "historical_provenance_references_preserved": 1.0,
            # Catalog rows preserved (~270 catalog table rows referencing
            # HNeRV/PR95 are HISTORICAL per Catalog #110/#113 APPEND-ONLY
            # and remain untouched). The audit-time count is approximate;
            # the canonical invariant is "HISTORICAL preserved" not exact
            # count.
            "catalog_rows_historical_references_preserved_approximate": 270.0,
        },
        recurrence_conditions=(
            "NEW design memo dated >= 2026-05-31 adopts 'HNeRV parity' as "
            "primary framing for binding-depth discipline",
            "NEW landing memo claims 'HNeRV parity discipline lesson X' as "
            "anchor rather than 'canonical leaderboard binding-depth lesson "
            "X' per CLAUDE.md canonical rename",
            "NEW canonical equation registration uses pr95_family_l<N>_*_vN "
            "prefix rather than canonical leaderboard binding-depth prefix "
            "(HISTORICAL prefixes preserved per Catalog #110/#113)",
            "NEW catalog row adopts 'HNeRV parity discipline' framing in "
            "row description rather than canonical binding-depth language",
            "subagent spawn prompt references 'HNeRV parity discipline' "
            "as primary framing for class-shift work without canonical "
            "rename context citation",
        ),
        canonical_source_anchor=(
            "operator binding META-correction 2026-05-30 verbatim 'any "
            "language like hnerv.or pr 95 parity language that might keep "
            "us stuck in local minima despite good intentions'; "
            ".omx/research/canonical_rename_wave_mapping_20260530.md D9 "
            "canonical anti-pattern specification; commit 6d3c42635 D1+D2 "
            "inline canonical rename wave landing; "
            "feedback_pr_or_greater_parity_synergy_binding_integration_"
            "not_hnerv_specific_meta_class_lesson_correction_20260530.md"
        ),
        canonical_unwind_path=(
            "Use 'canonical leaderboard binding-depth discipline' OR "
            "'PR-or-greater parity' OR 'binding-depth discipline' as "
            "primary forward-looking framing instead of 'HNeRV parity' or "
            "'PR95-family'; preserve historical references per Catalog "
            "#110/#113 APPEND-ONLY HISTORICAL_PROVENANCE (historical "
            "catalog rows + canonical equation prefix pr95_family_l<N>_*_v1 "
            "+ memory file cross-references all remain untouched); the "
            "canonical rename wave 2026-05-30 commit 6d3c42635 IS the "
            "canonical extinction event for the section-header surface; "
            "sister waves D3-D12 per canonical mapping memo extend to "
            "non-negotiable / STRICT-gate / catalog-extraction / council-"
            "roster surfaces."
        ),
        canonical_producers=(
            ".omx/research/*_design_*.md",
            "~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_*.md",
            "CLAUDE.md",
            "src/tac/canonical_equations/builtins.py",
            "src/tac/canonical_anti_patterns/builtins.py",
        ),
        canonical_consumers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
            "src/tac/canonical_anti_patterns.match_stack_against_anti_patterns",
            ".omx/research/canonical_rename_wave_mapping_20260530.md",
        ),
        paradigm_class=PARADIGM_RIGOR_LOSS,
        severity=SEVERITY_MEDIUM,
        provenance=_design_provenance(
            "hnerv_pr95_language_anchoring_local_minima_perpetuation_v1"
        ),
        empirical_falsifications=(
            EmpiricalFalsification(
                anti_pattern_id=(
                    "hnerv_pr95_language_anchoring_local_minima_perpetuation_v1"
                ),
                falsification_id=(
                    "canonical_rename_wave_d1_d2_inline_landing_commit_6d3c42635_20260530"
                ),
                measurement_method=(
                    "canonical rename wave D1+D2 inline landing per operator "
                    "blanket approval 'All are approved, land inline'; "
                    "section-header rename + HISTORICAL_PROVENANCE preservation "
                    "verified via git diff at commit 6d3c42635 (CLAUDE.md "
                    "section header line 186 + L14-L32 subsection header "
                    "line 214 renamed; ~270 historical catalog row references "
                    "preserved per Catalog #110/#113 APPEND-ONLY)"
                ),
                empirical_artifact_path=(
                    f"commit:{_D1_D2_RENAME_COMMIT_SHA}"
                ),
                empirical_output={
                    "extinction_event_type": "canonical_rename_wave_inline_landing",
                    "forward_looking_section_headers_renamed": 2,
                    "files_touched": 2,
                    "insertions": 186,
                    "deletions": 2,
                    "historical_provenance_references_preserved": True,
                    "historical_pr95_family_canonical_equation_prefix_preserved": True,
                    "historical_catalog_rows_preserved_approximate": 270,
                    "operator_approval_verbatim": "All are approved, land inline",
                    "extinction_surface_count": 2,
                    "sister_waves_d3_through_d12_queued": 10,
                },
                falsification_residual=0.0,
                captured_at_utc="2026-05-30T22:00:00Z",
                canonical_provenance=_design_provenance(
                    "hnerv_pr95_language_anchoring_local_minima_perpetuation_v1"
                ),
                incident_classification=(
                    INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE
                ),
                severity_observed=SEVERITY_OBSERVED_MEDIUM,
                operator_routable_unwind_path=(
                    "Sister waves D3-D12 per canonical mapping memo extend "
                    "the canonical rename + binding-depth discipline + 18-"
                    "assumption profile registration + 6-pillar canonical "
                    "helper landing surfaces across non-negotiable / STRICT-"
                    "gate / catalog-extraction / council-roster surfaces; "
                    "see .omx/research/canonical_rename_wave_mapping_"
                    "20260530.md for full D1-D12 plan"
                ),
            ),
        ),
        last_recalibration_utc=_D7_D8_D9_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


# -----------------------------------------------------------------------------
# Aggregator + idempotent registration
# -----------------------------------------------------------------------------


def build_all_d7_d8_d9_anti_patterns() -> list[AntiPattern]:
    """Return the 3 canonical rename wave D7+D8+D9 anti-patterns as a list.

    No registry write. Mirrors `tac.canonical_anti_patterns.builtins.
    build_all_initial_anti_patterns` API; suitable for canonical iteration
    + per-anti-pattern dispatch.
    """
    return [
        build_canonical_default_plateau_substrate_disguised_as_class_shift_v1(),
        build_micro_optimization_without_macro_escape_polishing_plateau_ceiling_v1(),
        build_hnerv_pr95_language_anchoring_local_minima_perpetuation_v1(),
    ]


def populate_d7_d8_d9_anti_patterns(
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> list[AntiPattern]:
    """Idempotent registration of the 3 canonical rename wave D7+D8+D9 anti-patterns.

    Per CLAUDE.md "Locked writes preserve deletions" (Catalog #132):
    APPEND-ONLY — re-running this helper appends new
    ``anti_pattern_registered`` events. The latest-row-wins query
    semantics in ``query_anti_patterns`` ensure consumers see the most
    recent payload. Mirrors `populate_initial_anti_patterns` API.

    Returns the 3 registered AntiPattern objects in canonical order:
    [D7, D8, D9].
    """
    out: list[AntiPattern] = []
    for ap in build_all_d7_d8_d9_anti_patterns():
        register_anti_pattern(
            ap,
            path=path,
            lock_path=lock_path,
            agent=agent,
            subagent_id=subagent_id,
            notes=(
                "D7+D8+D9 canonical rename wave registration per operator "
                "blanket approval 'All are approved, land inline' + "
                "canonical mapping memo .omx/research/canonical_rename_wave_"
                "mapping_20260530.md"
            ),
        )
        out.append(ap)
    return out


__all__ = [
    "build_all_d7_d8_d9_anti_patterns",
    "build_canonical_default_plateau_substrate_disguised_as_class_shift_v1",
    "build_hnerv_pr95_language_anchoring_local_minima_perpetuation_v1",
    "build_micro_optimization_without_macro_escape_polishing_plateau_ceiling_v1",
    "populate_d7_d8_d9_anti_patterns",
]
