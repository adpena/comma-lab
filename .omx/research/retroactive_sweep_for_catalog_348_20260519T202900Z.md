# Retroactive Sweep For Catalog #348 - Event-Driven Gate Sweep Evidence

Author: Codex
UTC: 2026-05-19T20:29:00Z
Catalog: #348
Gate: `check_new_gate_landing_includes_retroactive_sweep_evidence`
Source directive: `.omx/research/codex_routing_directive_session_20260519_max_score_lowering_batch_BCEF_20260519T051028Z.md`
Task: `codex_routing_directive_session_20260519_max_score_lowering_batch_BCEF_20260519T051028Z::CLUSTER_B`

## Bug-Class Symptom Signature

A new preflight gate lands and correctly prevents a future bug class, but no
retroactive sweep checks whether older KILL / DEFER / FALSIFY verdicts depended
on the same now-invalid evidence basis. The symptom is a stale verdict that
continues to influence lane registry, autopilot, or operator routing after the
underlying failure mode has been structurally fixed.

## Pre-Fix Window

Pre-fix window for this meta-gate: all `src/tac/preflight.py` gate landings
before Catalog #348 was committed with an enforced event-driven sweep contract.
Operationally, the initial implementation scans recent commits and runs
WARN-only so existing gates can be backfilled with
`retroactive_sweep_for_catalog_<N>_<utc>.md` memos or explicit waivers before a
future strict flip.

## Historical-KILL/DEFER/FALSIFY Search Results

Direct parent evidence:

- `.omx/research/meta_bug_retroactive_defer_kill_falsify_audit_20260519T044057Z.md`
  identified the stale-verdict class and recommended an event-driven sibling to
  the time-driven Catalog #300 retrospective discipline.
- `.omx/research/cable_c6_re_eval_high_symposium_drafts_synthesis_20260519T060557Z.md`
  shows the practical consequence: multiple resurrection candidates needed
  re-evaluation after later catalog gates invalidated the original rejection
  evidence.
- `.omx/research/codex_routing_directive_session_20260519_max_score_lowering_batch_BCEF_20260519T051028Z.md`
  promoted the meta-finding into Codex-owned Cluster B implementation work.

This initial sweep does not mark older findings KILL. It records that the
class is DEFER-pending-backfill for earlier gates and that future gate landings
must carry sweep evidence or a reviewed waiver.

## Per-Finding RE-EVAL-Priority Assignment

- HIGH: Any prior verdict whose memo names a bug class now covered by a newly
  landed gate and whose substrate has active or recently reactivated score
  potential. This includes the C6 resurrection queue and any KILL / DEFER /
  FALSIFY verdict later cited in paid dispatch or autopilot routing.
- MEDIUM: Historical apparatus-maintenance findings whose stale verdict would
  not affect immediate dispatch but could mislead future design synthesis.
- LOW: Purely archival or superseded memos where the lane has a newer
  authoritative verdict and no active dispatch, promotion, or budget decision.

## Authority And Wire-In

Catalog #348 is intentionally WARN-only at landing. The gate is complete enough
to expose missing sweep evidence, but strict mode requires a separate backfill
wave for recent gates. This preserves signal without retroactively breaking
ongoing partner work.

No score claim, rank claim, promotion claim, or dispatch claim is made here.
