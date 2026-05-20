# SPDX-License-Identifier: MIT
"""``tac.commit_safety`` — STAGING-surface protection helpers for the
multi-subagent edit/commit collision class.

Background — bug class context
──────────────────────────────
CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
non-negotiable + Catalog #314 ``check_no_subagent_files_touched_absorption_in_bare_commits``
DETECTS the bug class POST-COMMIT (the commit already landed; the gate
surfaces a warn-only operator alert).

This package PREVENTS the bug class at the STAGING surface by exposing a
canonical helper that ``tools/subagent_commit_serializer.py`` (and the
``/commit`` slash-command pre-hook wrapper at
``tools/check_sister_checkpoint_before_git_add.py``) call BEFORE the
``git add`` invocation reaches the working tree. If a sister subagent has
declared one of our intended-to-commit files in its ``files_touched``
checkpoint within the last 60 minutes, we ABORT (rc=8) or recommend
WAIT_AND_RETRY (rc=9) rather than silently package its in-flight work
under our commit body.

Sister of Catalog #117 / #157 / #174 / #216 / #230 / #248 / #289 / #302 /
#314 — together they extinct the multi-subagent edit/commit collision class
at EIGHT surfaces (edit-time-checkpoint #302 + edit-time-bulk-op #230 +
commit-time-pre-pre-lock #157 + commit-time-staged #216 + commit-time-
lock-arbitration #117 + post-resolution-residual-marker #248 + bare-
commit-absorbs-in-flight-files DETECT #314 + STAGING-surface PREVENT
[this package + Catalog #340]).

Memory: ``feedback_catalog_314_prevention_enhancement_landed_20260519.md``.
Lane: ``lane_catalog_314_prevention_enhancement_20260519``.
"""
from __future__ import annotations

from tac.commit_safety.sister_checkpoint_guard import (
    DEFAULT_LOOKBACK_MINUTES,
    EXEMPT_FILES,
    OVERRIDE_ENV_FLAG,
    OVERRIDE_ENV_RATIONALE,
    CorruptCheckpointError,
    SisterCheckpointVerdict,
    bare_override_attempted,
    check_files_against_sister_checkpoints,
    parse_override_env,
)
from tac.commit_safety.pre_write_sister_check import (
    DEFAULT_LOOKBACK_HOURS,
    STAND_DOWN_FILE_OVERLAP_THRESHOLD,
    SisterRecentlyLandedVerdict,
    check_sister_files_recently_landed,
)

__all__ = (
    "DEFAULT_LOOKBACK_MINUTES",
    "EXEMPT_FILES",
    "OVERRIDE_ENV_FLAG",
    "OVERRIDE_ENV_RATIONALE",
    "CorruptCheckpointError",
    "SisterCheckpointVerdict",
    "bare_override_attempted",
    "check_files_against_sister_checkpoints",
    "parse_override_env",
    # WAVE-3-PRE-WRITE-SISTER-ACTIVITY-CHECK-HELPER 2026-05-20: sister of
    # Catalog #340 at the PRE-WRITE / git-log surface.
    "DEFAULT_LOOKBACK_HOURS",
    "STAND_DOWN_FILE_OVERLAP_THRESHOLD",
    "SisterRecentlyLandedVerdict",
    "check_sister_files_recently_landed",
)
