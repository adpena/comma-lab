# Recursive NO-SIGNAL-LOSS protocol 2026-05-14

**Operator directive 2026-05-14 (verbatim, layered)**:
1. *"be very careful to ensure no signal loss"* (original)
2. *"and also need to recursively ensure no signal loss"* (this extension)

**Tag**: `research_only=true`; canonical persistence handoff per CLAUDE.md "Subagent coherence-by-default" mandatory pre-read of `.omx/research/*_directive_*` files dated within the last 24 hours.

This file EXTENDS the original `.omx/research/recovery_session_20260514_directive_absolute_no_signal_loss_20260514.md` (the 7-rule protocol) to apply RECURSIVELY across the FULL active subagent tree — including nested spawns. Every subagent MUST read BOTH directives on pre-flight.

## Active subagent tree (snapshot 2026-05-14 post C6-NEXT-WAVE + AUTH-EVAL-GATE + Z3 + TIME-TRAVELER + C1 dispatch)

**Level 0 (top-level parent)**: operator session.

**Level 1 (active, in-flight)**:
- `a155dcdabb5e1595d` RECOVERY-1 — D1 L2 INTEGRATION + Catalog #220 + NVDEC forensic anchor
- `a0ac89e2f1720c9f6` C6-NEXT-WAVE — C6 grammar to MDL ablation tool + MDL density proxy + C6 first-anchor smoke
- `a1ae242cd7e132dde` AUTH-EVAL-GATE-D4 — Catalog #221 STRICT gate + D4 timeout root-cause
- `a9c9d0c44a401eebc` Z3-BALLE-HYPERPRIOR — Step 1 staircase smoke + bolt-on substrate
- `ac9f6ed4323efebfc` TIME-TRAVELER-STAIRCASE — Z4 cooperative-receiver + Z5 predictive-coding L1
- `a5bfbb956b6db089a` C1-WORLD-MODEL-FOVEATION — Long-term campaign L1 scaffold

**Level 1+ (completed)**:
- `ade723fc2e7c5510e` RECOVERY-3 — per-trainer Tier B/C/D + XRAY Batch 6 (LANDED commit `00e811b4c` + `17aeaf148`)
- `ae42f94dcb44b1d43` RECOVERY-2 — C6 finish + Modal harvest (LANDED commit `3e4571c3a`)

**Level 2+ (nested spawns from any Level 1 subagent)**: SUBJECT TO THIS DIRECTIVE TRANSITIVELY.

## RECURSIVE EXTENSION — 4 new rules layered on top of the original 7

### Rule R1: Every nested spawn MUST inherit both directive files

When a Level 1 subagent spawns a Level 2 child, the Level 2 child's prompt MUST include verbatim references to:
- `.omx/research/recovery_session_20260514_directive_absolute_no_signal_loss_20260514.md` (the original 7-rule protocol)
- `.omx/research/recursive_no_signal_loss_protocol_20260514.md` (this file)

Both files MUST be in the Level 2 child's mandatory pre-flight read list. Failure to propagate the directive chain is a Level 1 contract violation, captured in the Level 2 child's checkpoint via the field `inherited_directives: ["original_7_rule", "recursive_extension"]`.

### Rule R2: Sister-subagent ownership disjointness is RECURSIVELY enforced

The Level 1 sister-subagent ownership map is:

| Subagent | OWNS (files; do not touch from outside) |
|---|---|
| RECOVERY-1 | `src/tac/substrates/d1_segnet_margin_polytope/*` + Catalog #220 gate function + CLAUDE.md non-negotiable section for #220 |
| C6-NEXT-WAVE | `src/tac/substrates/c6_e4_mdl_ibps/*` + `tools/mdl_scorer_conditional_ablation.py` (C6 grammar extension) + C6 smoke dispatch surface + `scripts/operator_authorize_substrate_c6_*` |
| AUTH-EVAL-GATE-D4 | Catalog #221 gate function + D4 root-cause memos + sister gates for CLI-flag-bug META class |
| Z3-BALLE | `src/tac/substrates/z3_balle_hyperprior_bolton/*` + `experiments/train_substrate_z3_balle_hyperprior_bolton.py` + recipe + remote driver |
| TIME-TRAVELER-STAIRCASE | `src/tac/substrates/z4_cooperative_receiver_loss/*` + `src/tac/substrates/z5_predictive_coding_world_model/*` + trainers + recipes |
| C1-WORLD-MODEL | `src/tac/substrates/c1_world_model_foveation/*` + trainer + recipe + remote driver |

A Level 2 child inherits the parent's ownership scope AND any disjoint surfaces it explicitly claims via checkpoint. **No Level 2 child may modify a file owned by a different Level 1 subagent.** Cross-tree edits MUST go through the parent → operator decision path.

### Rule R3: Checkpoint discipline is the canonical recursion anchor

Per CLAUDE.md "Mandatory crash-resume protocol" + Catalog #206, every subagent at every level MUST write to `.omx/state/subagent_progress.jsonl` (fcntl-locked, append-only). Recursive enforcement adds two required fields:

- `parent_id_or_session` — explicit parent subagent ID (chain to top-level session)
- `inherited_directives` — list of directive files this subagent's prompt referenced

When a subagent crashes (e.g., API error, rate limit, OOM), the resumer queries `tools/subagent_checkpoint.py read --latest-incomplete --parent-id-or-session <chain>` and inherits BOTH the work-in-progress AND the directive chain. **Signal-loss recovery is structural, not memory-based.**

### Rule R4: NO SCOPE OVERRIDE — recursive trust ladder

A Level 1 subagent CANNOT relax the protocol for its Level 2 children. A Level 2 child CANNOT relax for Level 3. The 7+4 = 11 total rules are inviolable at every recursion depth.

The trust ladder is RESTRICTIVE-MONOTONIC: each level can add more constraints but cannot remove any. Specifically:
- Level 1 may add scope restrictions (e.g., "you only edit module X")
- Level 2 must honor BOTH Level 1's scope AND the 11 base rules
- A subagent receiving a Level N prompt MUST verify the prompt is restrictive-monotonic vs the directive chain; if it isn't, refuse to start and surface to operator

This is the META-meta protection: even if a future operator prompt accidentally weakens the constraints, the receiving subagent refuses to proceed until the chain is repaired.

## RECURSIVE TRUST-BUT-VERIFY AUDIT PATTERN

Per CLAUDE.md "Trust but verify: an agent's summary describes what it intended to do, not necessarily what it did", the recursive audit pattern is:

**For every Level N subagent landing:**
1. Read its checkpoint chain (`tools/subagent_checkpoint.py read --parent-id-or-session <id>`)
2. Verify `inherited_directives` field includes BOTH directive files
3. Verify `git log --author "Claude" --since <subagent_start>` commits trace through the canonical serializer (Catalog #117)
4. Verify the subagent's work touched only the disjoint ownership scope (Rule R2)
5. Verify no `subagent_progress.jsonl` rows were OVERWRITTEN (append-only invariant; Catalog #131 + #128 + #206)

**If any audit step fails**: surface to operator as `recursive_signal_loss_candidate` in the landing memo. Do NOT silently absorb the failure.

## CONCRETE SIGNAL-LOSS RISKS RECURSIVELY ENFORCED

The original 7-rule directive identified these signal-loss risks. The recursive extension adds protection against EACH of them at every recursion depth:

| Risk | Original 7-rule defense | Recursive extension |
|---|---|---|
| Dirty file overwrite | `--expected-content-sha256` (Catalog #157) | At every level: predecessor + sibling sha snapshots, not just parent's |
| `.omx/research/` deletion | "Never delete untracked memos" | Even nested spawns refuse `rm` on research dir |
| `subagent_progress.jsonl` overwrite | "Append-only" | `parent_id_or_session` + `inherited_directives` are MANDATORY fields, not optional |
| Modal call abandonment | "HARVEST OR LOSE" | Each spawn level documents call_id chain in checkpoint |
| Sister-subagent file ownership | Rule 6 verbatim | Explicit ownership map enumerated above (Rule R2) |
| Comment-only contract | CLAUDE.md FORBIDDEN_PATTERNS | This file IS the contract; reading it (mandatory) IS enforcement |
| Untracked dirty state | "Read it first" + diff | Same applied at every recursion level |
| Cross-tree edit | Rule 6 (implicit) | Explicit refusal + escalation path (Rule R2 + R4) |

## OBSERVABILITY (the operator's visibility into the recursive tree)

Surface the active tree on demand via:

```bash
.venv/bin/python tools/subagent_checkpoint.py read --latest-incomplete \
    --tree-format \
    --output-jsonl .omx/state/subagent_tree_snapshot_$(date +%Y%m%dT%H%M%SZ).jsonl
```

The snapshot file is HISTORICAL_PROVENANCE per Catalog #113 (append-only, immutable). It enables the operator to audit at any point: "what was the full subagent tree at time T, and which directive chain did each node inherit?"

## CROSS-REFS

- CLAUDE.md "Subagent coherence-by-default — NON-NEGOTIABLE, HIGHEST EMPHASIS" (the canonical orchestration-without-orchestrator pattern; this directive is its applied instance)
- CLAUDE.md "Mandatory crash-resume protocol" (Catalog #206 IS the recursion anchor)
- CLAUDE.md "Subagent commits MUST use serializer — NON-NEGOTIABLE" (Catalog #117 + #157 + #174 + #216 protect against parallel-edit collisions at every level)
- CLAUDE.md "Forbidden artifact-lifecycle violations" (Catalog #113; the 4-kind taxonomy applies to `.omx/research/*_directive_*` files as HISTORICAL_PROVENANCE)
- CLAUDE.md "Long-burn score-lowering campaign default — NON-NEGOTIABLE, HIGHEST EMPHASIS" (the operator's "aggressively pursuing across class too" directive that triggered the 3 new across-class subagent dispatches; THIS protocol governs them recursively)
- `.omx/research/recovery_session_20260514_directive_absolute_no_signal_loss_20260514.md` (the original 7-rule sister directive)

## OPERATOR-ROUTABLE DECISIONS (3)

1. **STRICT preflight gate for recursive checkpoint discipline**: should a future Catalog # (claim via `tools/claim_catalog_number.py claim --commit-via-serializer`) refuse subagent commits whose checkpoint records lack the `parent_id_or_session` + `inherited_directives` fields? Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable, this IS the structural extension of Catalog #206 to the recursive case. (Recommend: yes, after a 7-day warn-only window for the 6 in-flight subagents to be updated.)
2. **Tree snapshot cadence**: how often should `tools/subagent_checkpoint.py read --tree-format` be run? Options: (a) per-operator-turn, (b) every 30 min via cron, (c) only at landing-time. (Recommend: c, to avoid noise; the audit is forensic, not real-time.)
3. **Directive-chain validation as a STRICT preflight check**: should there be a sister Catalog # that refuses any Level N+1 subagent prompt that does NOT cite both `.omx/research/*_directive_*` files in its mandatory pre-read list? This is a META-meta gate (refuses prompts that don't propagate the chain). (Recommend: defer to Phase 4 polish; the chain is currently human-driven by the parent agent and works adequately.)

Tagged `research_only=true`. NO score claims. NO GPU spend. All active subagents pick this up on next checkpoint cycle (per CLAUDE.md mandatory `.omx/research/*_directive_*` last-24-hours pre-read).
