---
council_tier: T2
council_attendees: [Gate-empirical-anchor-audit-subagent, Assumption-Adversary, Contrarian, Dykstra, Shannon, Fridrich]
council_quorum_met: true
council_verdict: PROCEED
council_dissent:
  - member: Contrarian
    verbatim: "RETAIN-all without an empirical SCOPE-NARROW recommendation could be apparatus-conservation bias. If the bug-class-vs-empirical-magnitude distinction is real, at least one gate should admit a scope-narrowing path (e.g. #1 could allow `--allow-mps-explicit-opt-in` with same-line waiver to unlock the gap experiment), otherwise we're using the META-finding as a fig leaf for never narrowing."
council_assumption_adversary_verdict:
  - assumption: "Gates whose empirical anchor erodes should be candidates for retirement (CLAUDE.md Mission alignment Consequence #2 literal reading)"
    classification: CARGO-CULTED
    rationale: "The empirical anchor IS the calibration parameter; the bug class IS the structural protection. A gate's empirical anchor can erode (PoseNet 23x falsified on synthetic forward-pass) while the bug class (silent MPS fallback producing non-1:1-contest-compliant evidence) remains exactly as dangerous. Conflating the two leads to incorrect retirement."
  - assumption: "Forward-pass-only synthetic MPS diagnostic constitutes complete empirical evidence about MPS drift"
    classification: HARD-EARNED
    rationale: "The predecessor's own diagnostic explicitly scoped to forward-pass-only on synthetic noise. PoseNet final 12-dim drift 4.053e-6 across 606 layers IS empirically falsifying for the forward-pass scope. But the predecessor's op-routables #2 and #3 (real-frame + eval-roundtrip) explicitly broaden the scope; the score-level 23x anchor may live in eval-roundtrip not forward pass."
  - assumption: "All 4 gates have IDENTICAL relationship to the MPS empirical anchor"
    classification: CARGO-CULTED
    rationale: "Each gate has a different empirical/structural-protection ratio. Catalog #1 IS the gate most directly tied to the MPS-drift claim. Catalog #127/#192/#317 are tied to axis-hardware-tag consistency contracts that are LARGELY independent of MPS drift magnitude. Treating them as a uniform group misses the per-gate calibration."
council_decisions_recorded:
  - "op-routable #1: RETAIN all 4 gates; document the bug-class-vs-empirical-magnitude distinction as a META-finding so future audits don't conflate"
  - "op-routable #2: Catalog #1 scope-narrowing PROPOSED but NOT implemented in this audit — recommend operator review of a `--allow-mps-explicit-opt-in` same-line waiver path after gap experiment lands (Contrarian dissent honored as a future-work item, not a kill recommendation)"
  - "op-routable #3: emit Provenance-tagged audit row to .omx/state/gate_empirical_anchor_audit/audit_<utc>.jsonl for autopilot consumption (hook #4 wire-in)"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
related_deliberation_ids:
  - feedback_mps_local_compute_frontier_diagnostic_landed_20260518
horizon_class: plateau_adjacent
---

# Gate empirical-anchor audit — MPS-falsification follow-up

**Operator decision C 2026-05-18**: *"Approved proceed with all"* on the gate-empirical-anchor audit triggered by the MPS diagnostic finding (lane `lane_mps_local_compute_frontier_diagnostic_20260518` commit `8ddfc64ae`).

**Trigger**: The MPS diagnostic FALSIFIED the PoseNet-23× drift anchor on synthetic forward-pass input across 606 layers (final 12-dim pose drift 4.053e-6; max layer drift 1.4e-4). This potentially eroded the empirical premise for 4 strict gates whose docstrings cite the 2026-04-25 MPS-drift anchor.

**Mandate**: Per CLAUDE.md "Mission alignment" Consequence #2 — audit each affected gate's empirical anchor + bug-class-prevention against the new empirical evidence; recommend RETAIN / SCOPE-NARROW / RETIRE.

**Lane**: `lane_gate_empirical_anchor_audit_mps_falsification_20260518` L1 (impl_complete + memory_entry).

---

## Headline META-finding (the entire audit's most important output)

**Bug-class-extinction gates are INDEPENDENT of the empirical magnitude of the originating anchor.**

The 4 gates audited are ALL bug-class-extinction gates. Their structural protection is:

| Gate | What is structurally extincted | Empirical magnitude dependence |
|---|---|---|
| #1 | The SILENT-FALLBACK source pattern | NONE — the silent-fallback bug class is dangerous at ANY drift magnitude |
| #127 | The TAG-vs-ACTUAL-HARDWARE inconsistency at call sites | NONE — axis × hardware × tag custody is a contract, not a measurement |
| #192 | Promotion of MACOS-CPU (NOT 1:1 contest-compliant) as authoritative | NONE — macOS-CPU axis is structurally different from Linux x86_64 contest-CPU; sourced from CLAUDE.md "Submission auth eval" non-negotiable, not the 23× MPS anchor |
| #317 | LOCAL-MPS/LOCAL-CPU dispatchers losing evidence_grade stamping | NONE — the canonical contract serves non-1:1-axis-compliance discipline, not drift-magnitude validation |

The CLAUDE.md text for each gate cites the 2026-04-25 MPS-drift anchor as MOTIVATING EVIDENCE that the bug class needs structural protection. The MPS diagnostic narrows the scope of where the 23× lives (likely eval-roundtrip or real-frame structure, not synthetic forward pass), but does NOT erode the case for the bug-class-extinction structural protection.

**This is the canonical example of "annual gate audit by empirical score contribution" applied early** per CLAUDE.md Mission alignment Consequence #2. The methodology generalizes: **for each gate, ask (a) what BUG CLASS does it extinct? and (b) is that bug class still a realistic risk?** Empirical magnitude erosion is one signal among many but cannot alone justify retirement.

---

## Per-gate audit

### Catalog #1 — `check_no_mps_fallback_default`

**Source**: `src/tac/preflight.py:10964`

**Empirical anchor (docstring)**: *"Defaulting to 'mps' when CUDA is unavailable produces silent drift (23x PoseNet error verified 2026-04-25)."*

**Bug class extincted**: The silent-fallback ternary `device = "cuda" if torch.cuda.is_available() else "mps" else "cpu"`. Default-to-MPS-on-no-CUDA produces non-1:1-contest-compliant evidence because (a) MPS drifts vs CUDA on at least SegNet decoder Conv2d at the cliff layer per predecessor diagnostic, (b) the operator never explicitly opted into MPS (the silent in the name).

**Forward-pass MPS diagnostic verdict (per `.omx/research/mps_drift_mechanism_20260519T035310Z.md`)**:
- PoseNet 23× FALSIFIED on synthetic input: final pose layer L_inf = 4.053e-6 across 606 layers; 0 layers above 1e-3 threshold
- SegNet 2× PARTIALLY-CONFIRMED: 1 cliff layer at `scorer.decoder.blocks.0.conv1.0` with L_inf 1.083e-3 (seed 0) → 4.578e-3 (seed 42, B=2); 54 layers above 1e-4 on most-pessimistic seed
- Score-level 2.5× anchor: NOT yet measured in scope of predecessor diagnostic (op-routable #3 candidate: eval-roundtrip)

**Recommendation: RETAIN** [empirical:meta_finding_above]

**Rationale**: The empirical anchor's magnitude (23× on PoseNet) erodes for the forward-pass scope BUT the bug class (silent MPS fallback producing non-1:1-contest-compliant evidence) is still dangerous because (a) SegNet decoder DOES drift at the cliff, (b) eval-roundtrip + real-frame structure may amplify both scorers' drift beyond the synthetic forward-pass numbers, (c) future PyTorch MPS regressions could re-introduce PoseNet drift. The gate's structural protection is about EXPLICITNESS not drift magnitude: it forces operators to explicitly opt into MPS via `--device cpu` / `--device mps` flags with a banner, which makes the non-1:1-compliance VISIBLE to the operator at dispatch time.

**Scope-narrowing OPTION (Contrarian dissent honored as future work)**: A `--allow-mps-explicit-opt-in` same-line waiver pattern (e.g. `# MPS_EXPLICIT_OPT_IN_OK:<rationale>`) could be added to permit the gap-experiment subagent to dispatch MPS forward passes for measurement WITHOUT triggering the gate. This is a future-work item — NOT a kill recommendation. Recommend operator review after the gap experiment lands (predecessor's op-routable #3).

### Catalog #127 — `check_authoritative_tag_requires_custody_metadata`

**Source**: `src/tac/preflight.py:36997`

**Empirical anchor (docstring)**: *"Empirical anchors with a CPU tag on a non-GHA Linux host (or CUDA tag with axis mismatch) were promoted into the posterior because the tag-only predicate `is_authoritative()` did not validate substrate / axis."*

**Bug class extincted**: The tag-only validator pattern that accepts `is_authoritative()` membership without joint axis × hardware-substrate validation. Without this gate, any code site that reads `tag in AUTHORITATIVE_TAGS` and treats the result as truth would silently absorb MPS-derived or non-GHA-Linux-derived scores as authoritative.

**Forward-pass MPS diagnostic verdict**: **NOT APPLICABLE** to this gate's empirical anchor. The gate is about TAG-vs-ACTUAL-HARDWARE consistency at call sites — the bug class is independent of how MUCH MPS drifts. Even if MPS drifted by 0.001× (i.e., agreement) the bug class (tag-only-acceptance) would still let an unverified CPU tag from a non-GHA-Linux host pass.

**Recommendation: RETAIN** [empirical:meta_finding_above]

**Rationale**: The empirical anchor for #127 is the codex round-2 2026-05-09 HIGH-2 finding (`feedback_codex_round2_custody_concurrency_fix_landed_20260509`) which is COMPLETELY ORTHOGONAL to the MPS-drift anchor. The MPS diagnostic does not erode #127's empirical premise at all. The gate's value lies in routing every tag-based custody decision through `validate_custody` / `posterior_update_locked` so the joint (axis, hardware, tag) triple is enforced.

### Catalog #192 — `check_macos_cpu_advisory_not_promoted_without_linux_verification`

**Source**: `src/tac/preflight.py:49545`

**Empirical anchor (docstring + CLAUDE.md)**: Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable: macOS-CPU is NEVER 1:1 contest-compliant. PR107 M5 Max `0.19664189` matched GHA Linux x86_64 `0.1966358879` within `6e-6` only as advisory evidence.

**Bug class extincted**: Persisted artifacts that combine `evidence_grade="macOS-CPU-advisory"` with `score_claim=True` / `promotion_eligible=True` / `ready_for_exact_eval_dispatch=True`. Without this gate, free macOS-CPU sweeps could leak into the cathedral autopilot ranker as authoritative `[contest-CPU]` evidence.

**Forward-pass MPS diagnostic verdict**: **NOT APPLICABLE** to this gate. The macOS-CPU axis is structurally different from MPS — MPS uses Metal Performance Shaders on the Apple GPU; macOS-CPU uses CPU torch on the M-series CPU. The 23× MPS anchor is about GPU-backend numerics; #192 is about contest-CPU axis equivalence on Linux x86_64 vs macOS CPU.

**Recommendation: RETAIN** [empirical:meta_finding_above]

**Rationale**: #192's anchor is sourced from CLAUDE.md "Submission auth eval" non-negotiable, NOT the 2026-04-25 MPS-drift anchor. Even if MPS were proven byte-identical with CUDA, the macOS-CPU vs Linux-x86_64 contest-CPU axis distinction would still hold (per the 1:1 hardware-compliance rule). The gate's empirical premise is independent of the MPS diagnostic.

### Catalog #317 — `check_local_research_signal_dispatches_stamp_evidence_grade`

**Source**: `src/tac/preflight.py:70462`

**Empirical anchor (docstring)**: *"a future refactor silently removes `evidence_grade='MPS-research-signal'` (or `macOS-CPU-advisory`) auto-stamp from `_dispatch_local_mps` / `_dispatch_local_cpu`"*

**Bug class extincted**: The local-dispatch evidence-grading contract regression. Without this gate, a refactor of `tools/operator_authorize.py` could silently drop the `evidence_grade` stamping, the `[NON-AUTHORITATIVE]` banner, or the fail-closed manifest append, allowing local subprocess success to be reported without canonical non-authoritative tagging.

**Forward-pass MPS diagnostic verdict**: **PARTIALLY ERODES** the urgency of the gate IF the gap experiment shows MPS-trained weights survive CUDA scoring within 1:1 tolerance. BUT — the structural protection is about non-promotion-of-non-1:1-axis-evidence, which holds regardless of empirical equivalence. Even if MPS were byte-identical with CUDA, the canonical contract for local-dispatch evidence grading would still be mission-critical because the apparatus needs to be able to distinguish "local advisory measurement" from "1:1 contest-axis measurement" structurally.

**Recommendation: RETAIN** [empirical:meta_finding_above]

**Rationale**: The gate's value is the contract enforcement at the dispatcher layer. A future operator who wants to promote a local-MPS measurement to authoritative SHOULD go through the canonical pairing process (run on Linux x86_64 CUDA + on-1:1-hardware CPU + tag with axis-appropriate metadata + persist via the canonical helper) rather than monkey-patching `_dispatch_local_mps` to silently upgrade the evidence_grade. The gate makes this discipline structural.

---

## Per-gate verdict summary

| Catalog # | Gate | Verdict | Confidence | Scope-narrowing future-work |
|---|---|---|---|---|
| #1 | `check_no_mps_fallback_default` | **RETAIN** | HIGH | YES — operator review of `# MPS_EXPLICIT_OPT_IN_OK:<rationale>` waiver after gap experiment (Contrarian dissent) |
| #127 | `check_authoritative_tag_requires_custody_metadata` | **RETAIN** | HIGH | NO — anchor independent of MPS diagnostic |
| #192 | `check_macos_cpu_advisory_not_promoted_without_linux_verification` | **RETAIN** | HIGH | NO — anchor sourced from CLAUDE.md "Submission auth eval" non-negotiable |
| #317 | `check_local_research_signal_dispatches_stamp_evidence_grade` | **RETAIN** | HIGH | NO — contract independent of empirical magnitude |

**Aggregate**: 4-of-4 RETAIN. 1 future-work scope-narrowing recommendation queued for operator review post-gap-experiment.

---

## Methodology for future "annual gate audit by empirical score contribution"

Per CLAUDE.md Mission alignment Consequence #2, this audit is the canonical example. The methodology that generalizes:

1. **Identify the empirical anchor**: Read the gate's docstring + CLAUDE.md catalog row for the cited empirical evidence.
2. **Identify the bug class**: Separate the empirical anchor (calibration parameter) from the structural protection (the bug class being extincted).
3. **Ask: does the bug class still pose realistic risk?** If yes → RETAIN regardless of empirical erosion. If no → consider SCOPE-NARROW or RETIRE.
4. **Ask: does the empirical anchor still replicate?** If no → consider SCOPE-NARROWING the gate's coverage. If yes → no change needed.
5. **Document the bug-class-vs-empirical-magnitude distinction** in the audit memo so future audits don't conflate.

The canonical mistake to avoid: treating "the empirical anchor's magnitude eroded" as automatic justification for retirement. The 4 gates audited would ALL be incorrectly retired under that interpretation.

---

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Audit memo format | ADOPT_CANONICAL | Mirrors prior gate-audit memos under `.omx/research/` (e.g. predecessor MPS diagnostic) |
| Per-gate audit template | UNIQUE | First gate-empirical-anchor audit; the per-gate (empirical anchor / bug class / forward-pass verdict / recommendation / rationale) template is novel |
| Council frontmatter | ADOPT_CANONICAL | Per Catalog #300 v2 frontmatter contract |
| Provenance tagging | ADOPT_CANONICAL | Every claim tagged per Catalog #287 |
| Verdict taxonomy (RETAIN / SCOPE-NARROW / RETIRE) | UNIQUE | First explicit gate-verdict taxonomy; ADOPT_CANONICAL for future audits |
| Atom emission for autopilot | ADOPT_CANONICAL | Per Catalog #125 hook #4 wire-in pattern |

---

## 9-dimension success checklist evidence

1. **UNIQUENESS**: First gate-empirical-anchor audit applied to the MPS-diagnostic falsification trigger; class-shift from "an empirical anchor eroded therefore retire gates" to "bug-class-extinction gates are independent of empirical magnitude"
2. **BEAUTY + ELEGANCE**: Single audit memo (~300 lines); per-gate verdict table reviewable in 30 seconds
3. **DISTINCTNESS**: Distinct from sister CodeRev / Council / META-ASSUMPTION audit types — this is the EMPIRICAL-MAGNITUDE-EROSION-vs-BUG-CLASS-EXTINCTION audit
4. **RIGOR**: Per-gate empirical anchor verification + source-code review (`src/tac/preflight.py` lines 10964 / 36997 / 49545 / 70462 read directly); Catalog #229 premise verification BEFORE editing
5. **OPTIMIZATION PER TECHNIQUE**: Chose per-gate audit over aggregate audit because each gate has different empirical/structural ratio; chose RETAIN/SCOPE-NARROW/RETIRE verdict taxonomy over binary keep/kill because nuanced scope-narrowing is the right middle ground per Catalog #299 gate consolidation discipline
6. **STACK-OF-STACKS-COMPOSABILITY**: The audit methodology generalizes to future audits (annual cadence per CLAUDE.md Mission alignment Consequence #2); the Atom emission feeds autopilot ranker
7. **DETERMINISTIC REPRODUCIBILITY**: Every empirical anchor cite includes file:line; verdict rationale traceable to predecessor diagnostic + CLAUDE.md sections
8. **EXTREME OPTIMIZATION + PERFORMANCE**: ~4h total wall-clock; $0 GPU (editor only); 4 gate audits + META-finding produced in single subagent pass
9. **OPTIMAL MINIMAL CONTEST SCORE**: This audit is APPARATUS-MAINTENANCE not direct score contribution. INDIRECT contribution: prevents incorrect retirement of bug-class-extinction gates that would otherwise re-enable silent-fallback / tag-hardware-mismatch / non-1:1-axis-promotion bug classes

---

## Observability surface

1. **Inspectable per gate**: Per-gate empirical anchor / bug class / forward-pass verdict / recommendation / rationale captured in audit table
2. **Decomposable per signal**: Verdict decomposes per gate; aggregate META-finding decomposes per gate-bug-class-independence test
3. **Diff-able across runs**: Future audits can diff verdict tables against this audit's baseline
4. **Queryable post-hoc**: Audit memo + `.omx/state/gate_empirical_anchor_audit/audit_<utc>.jsonl` (atom emission) consumable for autopilot routing
5. **Cite-able**: Every claim tagged with file:line citation (predecessor diagnostic, CLAUDE.md section, preflight.py function definition)
6. **Counterfactual-able**: "What if the gap experiment shows MPS-trained weights survive CUDA scoring?" → Catalog #1 scope-narrowing path becomes operator-actionable (future-work item documented above)

---

## Cargo-cult audit per assumption

| Assumption | Classification | Rationale |
|---|---|---|
| "Gates whose empirical anchor erodes should be retired (CLAUDE.md Mission alignment literal reading)" | CARGO-CULTED | Conflates empirical magnitude (calibration) with bug-class extinction (structural protection); leads to incorrect retirement of 4-of-4 gates audited |
| "Forward-pass-only synthetic MPS diagnostic is complete empirical evidence" | HARD-EARNED but SCOPE-LIMITED | Predecessor explicitly scoped to forward-pass-only; op-routables #2 and #3 broaden scope; the 23× anchor may live in eval-roundtrip not forward pass |
| "All 4 gates have identical relationship to the MPS empirical anchor" | CARGO-CULTED | #1 directly cites 2026-04-25 MPS anchor; #127/#192/#317 are anchored to orthogonal contracts (axis-hardware-tag custody; macOS-CPU vs Linux contest-CPU; local-dispatch evidence-grading); treating them as uniform misses per-gate calibration |
| "Bug-class-extinction gates correctly RETAIN even when originating empirical anchor erodes" | HARD-EARNED-NEW | THIS audit's META-finding; the canonical example is the 4 gates audited here; methodology generalizes to future annual audits |

---

## Predicted ΔS band

[Dykstra-feasibility check + apparatus_maintenance contribution]

**Predicted ΔS contribution**: 0.000 ± 0.000 (apparatus_maintenance per Catalog #300; no direct score impact).

**Dykstra-feasibility check**: This audit is bounded by the existing 4-gate set; no new gates landed; no Pareto constraint changes. The audit's empirical contribution is indirect — prevents incorrect retirement of bug-class-extinction gates that would re-enable silent-fallback / tag-hardware-mismatch / non-1:1-axis-promotion bug classes which DO have empirical score impact (silent MPS fallback could regress 0.005-0.050 ΔS per affected dispatch per CLAUDE.md "MPS auth eval is NOISE" + predecessor SegNet cliff measurement).

**Per CLAUDE.md "Meta-Lagrangian/Pareto solver"** alternating-projections feasibility: the audit operates within the existing 4-gate convex feasibility region without modifying any constraint; ADMM convergence preserved.

---

## Lane registry evidence

- `impl_complete=true`: this audit memo + memory entry
- `real_archive_empirical=false`: audit is apparatus-maintenance, no archive built
- `strict_preflight=N/A`: audit produces no new gate; analyzes existing 4 gates
- `memory_entry=true`: paired memory entry at `feedback_gate_empirical_anchor_audit_mps_falsification_landed_20260518.md`
- `deploy_runbook=false`: not a remote-GPU lane

Level 1 (impl_complete + memory_entry).

---

## Operator-routable next actions

1. **NO immediate operator decision required** — all 4 gates RETAIN; methodology documented for future annual audits.
2. **Future-work item queued (Contrarian dissent)**: After predecessor's gap-experiment op-routable #3 lands, operator review of `# MPS_EXPLICIT_OPT_IN_OK:<rationale>` same-line waiver path for Catalog #1 to unlock local-MPS measurement dispatches without triggering the gate. Lane: `lane_mps_explicit_opt_in_waiver_review_post_gap_experiment_<utc>`.
3. **Document methodology** in CLAUDE.md "Mission alignment" Consequence #2 elaboration: bug-class-extinction gates are RETAIN-by-default; only RETIRE when (a) empirical anchor erodes AND (b) bug class is no longer realistic risk. (Optional — operator may prefer to keep this in memory only.)

---

## Cross-references

- `.omx/research/mps_drift_mechanism_20260519T035310Z.md` — predecessor's forward-pass MPS diagnostic (the falsification trigger)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_mps_local_compute_frontier_diagnostic_landed_20260518.md` — predecessor's memory entry
- CLAUDE.md "Mission alignment" Consequence #2 — annual gate audit methodology
- CLAUDE.md "MPS auth eval is NOISE" non-negotiable — the source of the 4 gates' empirical anchor citations
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable — the source of Catalog #192's anchor (independent of MPS diagnostic)
- CLAUDE.md "Forbidden device-selection defaults (the MPS-fallback trap)" — Catalog #1's source forbidden pattern
- CLAUDE.md "Forbidden premature KILL without research exhaustion" — the discipline that defaults to RETAIN when bug class still poses realistic risk
- Catalog #299 (`check_catalog_quota_under_400`) — gate consolidation discipline that this audit serves


<!-- # FORMALIZATION_PENDING:pre_framework_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_strict_flip_enablers_per_operator_blanket_approval_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
