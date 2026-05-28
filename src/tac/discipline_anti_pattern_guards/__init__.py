# SPDX-License-Identifier: MIT
"""Canonical helpers for SPAWN-time + HANDOFF-time discipline anti-patterns.

Sister of ``tac.commit_safety`` (Catalog #340 STAGING-time guard). Where
``commit_safety.check_files_against_sister_checkpoints`` extincts the
bare-commit-absorbs-in-flight-files class at the staging surface, this package
extincts two upstream discipline classes at the SPAWN-time + POST-SPAWN-HANDOFF
surfaces:

* **Anti-pattern #13** ``subagent_spawn_without_head_state_premise_verification_v1``
  — parent agent invokes ``Agent`` spawn without first running
  ``git log --oneline -30`` + ``git status`` + sister-landing-memo check.
  Empirical receipts: 2 STAND_DOWN incidents in Wave N+5 (Slot 1 Compound C
  predecessor commit ``e61ea93b0`` + Slot 2 framework_agnostic STAND_DOWN
  resolved at ``5d38bf9df``). Canonical unwind per Catalog #229 (premise-
  verification-before-edit). Severity ``medium_substrate_regression``;
  ``is_actively_recurring=True`` (>=2 falsifications).

* **Anti-pattern #14** ``predecessor_working_tree_uncommitted_handoff_v1`` —
  predecessor subagent's ``SUBAGENT_TERMINATE`` / ``STAND_DOWN`` leaves
  files uncommitted in the shared working tree; successor inherits a dirty
  state that is hard to distinguish from sister-subagent in-flight
  collision (Catalog #314 absorption-pattern). Resolution: predecessor's
  canonical-serializer auto-commit OR explicit ``SUPERSESSION-PENDING``
  declaration. Severity ``low_implementation_inefficiency``; 1 receipt.

Public API
──────────
* :class:`SpawnGuardVerdict` — typed verdict for spawn-time PV.
* :func:`verify_head_state_before_spawn` — canonical sister of Catalog #229
  PV at agent-spawn surface; returns ``PROCEED`` / ``DUPLICATE_HEAD_STATE``
  / ``SISTER_IN_FLIGHT``.
* :class:`HandoffGuardVerdict` — typed verdict for predecessor handoff.
* :func:`verify_predecessor_working_tree_committed_or_auto_commit` —
  consumes canonical serializer; auto-commits via
  ``tools/subagent_commit_serializer.py`` if predecessor uncommitted
  work detected (post-handoff hook).

Wire-in surfaces
────────────────
1. ``src/tac/preflight.py`` Catalog #376
   ``check_subagent_spawn_includes_head_state_pv_evidence`` — STRICT
   preflight gate (WARN-ONLY initial wire-in per "Strict-flip atomicity
   rule"; live-repo backfill sweep then strict-flip).
2. ``tools/subagent_checkpoint.py`` (downstream consumer; predecessor-
   resume helper already exists per Catalog #206 sister discipline).
3. ``tools/operator_authorize.py`` (downstream consumer for agent-spawn
   wrappers when they exist).

Per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: this package is
NEW; no mutation of sister registries. Per CLAUDE.md "Beauty, simplicity,
and developer experience": narrow typed API; ``@dataclasses.dataclass(frozen=True)``
verdicts; no hidden state; every recommendation enumerable.

Sister of:
    Catalog #229 (premise-verification-before-edit) — design-memo surface
    Catalog #117 (subagent commit serializer must be used) — last-50-commit surface
    Catalog #157/#174/#216/#289 (commit-swap family) — commit-time surfaces
    Catalog #302 (sister subagent scope overlap via checkpoint JSONL) — edit-time surface
    Catalog #314 (bare-commit absorbs in-flight files) — POST-COMMIT surface
    Catalog #340 (sister-checkpoint guard at STAGING surface) — STAGING-time surface
    Catalog #376 (THIS gate's strict preflight) — SPAWN-time surface
    Catalog #344 (canonical equations registry) — formalization surface

Together they extinct multi-subagent edit/commit/spawn collision class
across NINE surfaces:
    spawn-time (#374) + edit-time-checkpoint (#302) + edit-time-bulk-op (#230)
    + commit-time-pre-pre-lock (#157) + commit-time-staged (#216)
    + commit-time-lock-arbitration (#117 + #174)
    + post-resolution-residual-marker (#248)
    + post-commit-absorption-detect (#314)
    + staging-surface-prevent (#340)

Lane: lane_strict_gate_check_subagent_spawn_includes_head_state_pv_evidence_plus_canonical_helper_20260528
Memory: feedback_strict_gate_check_subagent_spawn_includes_head_state_pv_evidence_plus_canonical_helper_landed_20260528.md
"""
from tac.discipline_anti_pattern_guards.subagent_spawn_head_pv_guard import (
    DEFAULT_SISTER_LANDING_LOOKBACK_MINUTES,
    DEFAULT_SISTER_LANDING_MEMO_GLOB,
    HEAD_STATE_PV_TOKENS,
    SpawnGuardRecommendation,
    SpawnGuardVerdict,
    SpawnPvEvidenceContext,
    verify_head_state_before_spawn,
)
from tac.discipline_anti_pattern_guards.predecessor_handoff_auto_commit_guard import (
    HandoffGuardRecommendation,
    HandoffGuardVerdict,
    PredecessorHandoffContext,
    verify_predecessor_working_tree_committed_or_auto_commit,
)
from tac.discipline_anti_pattern_guards.main_thread_spawn_decision_pv_guard import (
    DEFAULT_LOOKBACK_MINUTES as MAIN_THREAD_DEFAULT_LOOKBACK_MINUTES,
    DEFAULT_RECENT_COMMIT_LIMIT,
    WAIT_AND_RETRY_THRESHOLD_MINUTES,
    MainThreadSpawnGuardVerdict,
    MainThreadSpawnRecommendation,
    verify_head_state_before_main_thread_spawn,
)

__all__ = [
    # Spawn-time PV guard (anti-pattern #13 extinction; Catalog #376
    # SUBAGENT-side first-checkpoint surface).
    "DEFAULT_SISTER_LANDING_LOOKBACK_MINUTES",
    "DEFAULT_SISTER_LANDING_MEMO_GLOB",
    "HEAD_STATE_PV_TOKENS",
    "SpawnGuardRecommendation",
    "SpawnGuardVerdict",
    "SpawnPvEvidenceContext",
    "verify_head_state_before_spawn",
    # Predecessor handoff guard (anti-pattern #14 extinction).
    "HandoffGuardRecommendation",
    "HandoffGuardVerdict",
    "PredecessorHandoffContext",
    "verify_predecessor_working_tree_committed_or_auto_commit",
    # Main-thread spawn-decision PV guard (Catalog #378 PARENT-side
    # surface; sister of Catalog #376 at the upstream parent-agent
    # decision boundary; extincts the cascade-of-STAND_DOWNs bug class
    # at the SPAWN-decision time rather than the post-hoc detection
    # surfaces #314/#340).
    "MAIN_THREAD_DEFAULT_LOOKBACK_MINUTES",
    "DEFAULT_RECENT_COMMIT_LIMIT",
    "WAIT_AND_RETRY_THRESHOLD_MINUTES",
    "MainThreadSpawnGuardVerdict",
    "MainThreadSpawnRecommendation",
    "verify_head_state_before_main_thread_spawn",
]
