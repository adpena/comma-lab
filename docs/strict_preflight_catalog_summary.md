# Strict preflight catalog summary

Browseable summary of the strict-preflight catalog gates currently active in this repository.

The catalog is a set of structural bug-class extinction checks: each gate is a static or repo-state check that refuses to allow a code/state pattern that empirically caused a problem in a prior session. When a check fails, the operation (commit, dispatch, smoke, deploy) is blocked until the pattern is fixed or an explicit waiver is added.

Related: [`docs/asymptotic_floor_candidate_inventory.md`](asymptotic_floor_candidate_inventory.md) Section E.1. Sister library: [`adpena/tac`](https://github.com/adpena/tac). Submission packet: [`commaai/comma_video_compression_challenge#110`](https://github.com/commaai/comma_video_compression_challenge/pull/110).

---

## What it is

A registry of ~300 `check_*` gate functions, each scoped to one empirically-encountered bug class. Gates run via `python -m tac.preflight` in two modes: STRICT (fail-closed, raises a `PreflightError` on any violation; wired into commit hooks and dispatch wrappers) and WARN (logs violations, allows operation to proceed; used during the multi-commit window between landing a new gate and clearing the legacy backlog).

Gate signatures follow a convention: `check_<short_description>(strict: bool = False, verbose: bool = False) -> list[str]` returning a list of violation messages. Lifecycle is documented in CLAUDE.md per the "promotion path" pattern (a new gate starts `strict=False`, the live violation count is driven to 0 across the repo, then it is flipped `strict=True` in the orchestrator).

The lineage of this discipline traces to Rudin's interpretable-ML practice — *Stop Explaining Black Box Machine Learning Models for High Stakes Decisions* (Nat Mach Intell, 2019) is the canonical reference for why structural extinction of failure modes via interpretable static checks beats post-hoc explanation of opaque debugging sessions. The catalog table is operationally a falling rule list per Wang & Rudin (AISTATS 2015): higher-priority rules (commit-machinery + state-discipline) come first; lower-priority rules (substrate-discipline + production-hardening) follow; first-match-wins on rule fire.

## The discipline

Every adversarial-review finding that surfaces a real bug must be addressed with **two landings**, not one:

1. **The fix** — patch the immediate code surface that produced the bug.
2. **A STRICT preflight gate** — a `check_<bug_class_name>` function in the orchestrator module that refuses any code surface in the repo that re-introduces the bug class.

This is non-negotiable per CLAUDE.md "Bugs must be permanently fixed AND self-protected against." Single-surface fixes are insufficient because empirically the same bug class has a 6–7x spread across the repo: a fix at one surface leaves the same class active at six others. The dedicated gate forces the structural extinction at all surfaces.

The canonical example is the META-meta discovery 2026-05-08: the proactive sweep that originally identified the spread pattern found that a single fix to the custody-validator at one call site left the same validator-bypass class active at dozens of sister call sites. The structural fix was to land a gate that scans all call sites of the relevant family.

## Categories

The gates partition naturally into orthogonal surfaces. A representative sample (one or two gates per category) follows; the full table lives in `CLAUDE.md` under "Meta-bug class catalog (strict-mode preflight)."

| Category | Example gate | What it prevents |
|---|---|---|
| **Commit-machinery** | `check_subagent_commit_serializer_uses_lock` (#117) | Bare `git commit` outside the canonical serializer, which causes commit-body swap under concurrent subagent commits. |
| **State-discipline** | `check_no_bare_writes_to_shared_state` (#131) | Bare writes to `.omx/state/*.jsonl` outside the canonical fcntl-locked helper, which silently drops concurrent updates. |
| **Substrate-discipline** | `check_substrate_at_optimal_form_before_paid_dispatch` (#315) | Dispatching a substrate for a paid Modal/Lightning/Vast.ai anchor when the latest council deliberation returned PROCEED_WITH_REVISIONS, which empirically falsifies the implementation but mislabels as paradigm-kill. |
| **Council-discipline** | `check_grand_council_deliberation_has_explicit_assumption_statements` (#292) | Council memos that reach consensus without per-member assumption surfacing, which empirically produces assumption-bound consensus that misses the real frontier-breaking move. |
| **Dispatch-discipline** | `check_dispatch_optimization_protocol_complete` (#270) | Paid dispatch with a substrate trainer + recipe + lane driver state that fails the Tier 1 / Tier 2 / Tier 3 production-hardening protocol, which empirically caused 6+ dispatch failures in a 24h window at $0.50–$15 each. |
| **Production-hardening** | `check_modal_dispatcher_uses_canonical_mount_builder` (#153) | Hand-curated Modal mount lists that drift out of sync with the trainer's actually-required input files, which empirically caused a $0.016 / 15s OOM-on-missing-input crash. |
| **Post-training validation** | `check_no_predicted_band_without_post_training_tier_c_validation` (#324) | Substrate recipes claiming a predicted ΔS band from random-init Tier-C density, which empirically produced a 22x miss (predicted `[0.113, 0.163]` vs actual `3.04` on the C6 IBPS information-bottleneck substrate). |

## The META-meta gates

A small subset of gates protect the catalog itself from drift:

- `check_claude_md_catalog_no_duplicate_numbers` (#118) — refuses CLAUDE.md catalog table entries with duplicate numbers.
- `check_claude_md_catalog_text_matches_preflight_strict_value` (#159) — refuses CLAUDE.md catalog entries whose strictness text contradicts the `strict=` value wired in the orchestrator.
- `check_strict_preflight_callsites_have_claude_md_catalog_row` (#176) — refuses STRICT-flipped gates that lack a matching numbered row in the CLAUDE.md catalog table.
- `check_strict_flipped_catalog_entries_have_live_count_zero` (#185) — refuses CLAUDE.md catalog entries that claim "Live count: 0" while the underlying gate function actually returns positive violations (extincts the drift between documentation and empirical state).
- `check_catalog_claim_committed_via_serializer` (#186) — refuses bare `tools/claim_catalog_number.py` invocations that do not flow through the canonical git-transactional serializer.
- `check_catalog_quota_under_400` (#299) — refuses catalog table entries above the #400 quota without explicit operator review (the gate-consolidation discipline brake, designed to force a "stop and consolidate" pause when the catalog approaches structural-overhead saturation rather than mechanically growing without audit).

Together these six gates extinct the failure modes where the catalog itself accumulates phantom rows, drifts out of sync with the orchestrator, or grows without consolidation review.

## Anti-pattern this prevents

The dominant failure mode the catalog discipline extincts is **per-instance fixes that leave the META class active at sister surfaces**.

Concretely: an adversarial review surfaces a bug; the operator patches the cited file:line; ships the fix; declares the bug closed. Six commits later, the same bug class surfaces at a sister call site that the original fix did not touch. The cost is paid twice (once at the original surface, once at the sister), and the second cost is harder to attribute because the failure mode looks different (different file, different error message, different downstream symptom) even though the root cause is identical.

The dedicated gate forces the operator to scan the entire repo for the bug-class pattern at fix time, not retroactively after the second instance hits. The bug class becomes structurally extinct rather than per-instance suppressed.

## Honest scope

The catalog is engineering rigor, not a contest-score primitive by itself. It does not produce score reductions directly; it reduces the cost of the substrate-design loop by extincting failure modes that would otherwise consume GPU spend, wall-clock, and operator attention on debugging cycles that the static checks could have caught at commit time.

The cost of the discipline is real: ~300 gates means ~300 maintenance surfaces, each with its own waiver mechanism, edge cases, and false-positive risk. The META-meta gates (#118 / #159 / #176 / #185 / #186 / #299) exist precisely because the catalog's own maintenance overhead is itself a bug-class surface. The quota brake at #400 is a structural commitment to pause and consolidate before the maintenance cost crosses the value the gates provide.

Sister library [`adpena/tac`](https://github.com/adpena/tac) carries a subset of the canonical helpers the preflight gates protect (the fcntl-locked JSONL helpers, the canonical Modal call_id ledger, the master-gradient extraction primitives, the substrate registry contract). The catalog gates and the sister library are co-developed: each new canonical helper lands paired with the STRICT gate that protects callers from bypassing it.
