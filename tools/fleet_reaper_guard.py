#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed launch guard for the operator's fleet process reaper.

The fleet reaper selects detached/no-TTY processes whose *visible command
line* contains a standalone ``claude`` or ``codex`` word, after excluding
intentional-daemon command lines containing one of its exemption markers.
This module models that command-line leg of the source predicate.  The other
legs (age, TTY, parent/stdin state) become true over time for an ordinary
detached launch, so canonical launchers must close this leg before spawning.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REAPER_KEEPALIVE_TOKEN = "REAPER_KEEPALIVE"
REAPER_NAME_PREDICATE = re.compile(r"\b(claude|codex)\b")
REAPER_EXEMPTION_PREDICATE = re.compile(
    r"codex_runs/|REAPER_KEEPALIVE|/Applications/[^ ]*\.app/"
)
_ENV_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*", re.DOTALL)


@dataclass(frozen=True)
class FleetReaperAssessment:
    """Source-equivalent command-line facts for one stable process image."""

    requested_argv: tuple[str, ...]
    stable_argv: tuple[str, ...]
    matched_tokens: tuple[str, ...]
    matching_argv_parts: tuple[str, ...]
    exemption_hits: tuple[str, ...]

    @property
    def refused(self) -> bool:
        return bool(self.matched_tokens and not self.exemption_hits)

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_argv": list(self.requested_argv),
            "stable_argv": list(self.stable_argv),
            "matched_tokens": list(self.matched_tokens),
            "matching_argv_parts": list(self.matching_argv_parts),
            "exemption_hits": list(self.exemption_hits),
            "refused": self.refused,
            "source_predicate_scope": (
                "command-line name/exemption leg; detached launch supplies the "
                "no-TTY/dead-stdin-or-orphan/age legs over time"
            ),
        }


class FleetReaperLaunchRefusal(OSError):
    """Raised before spawn when a stable detached argv enters the reap set."""

    def __init__(self, assessment: FleetReaperAssessment) -> None:
        self.assessment = assessment
        super().__init__(
            "stable argv matches the fleet reaper predicate without an exact exemption: "
            + " ".join(assessment.matching_argv_parts)
        )


def stable_process_argv(argv: Sequence[str]) -> tuple[str, ...]:
    """Return the argv expected after leading ``env KEY=VALUE`` exec wrappers.

    ``/usr/bin/env`` does not remain as the long-lived process: it installs
    variables and execs its target.  Consequently, putting
    ``REAPER_KEEPALIVE=1`` only in that transient prefix does *not* prove that
    the marker remains visible to ``ps``.  Strip simple leading env wrappers
    recursively and assess the process image that the reaper will actually
    inspect.  Unknown env option shapes are left intact and therefore handled
    conservatively by the normal predicate.
    """

    stable = tuple(str(part) for part in argv)
    while stable and Path(stable[0]).name == "env":
        index = 1
        if index < len(stable) and stable[index] == "--":
            index += 1
        while index < len(stable) and _ENV_ASSIGNMENT.fullmatch(stable[index]):
            index += 1
        if index >= len(stable) or (index < len(stable) and stable[index].startswith("-")):
            break
        stable = stable[index:]
    return stable


def assess_detached_argv(argv: Sequence[str]) -> FleetReaperAssessment:
    """Assess the exact stable command line against the fleet source regexes."""

    requested = tuple(str(part) for part in argv)
    stable = stable_process_argv(requested)
    matching_parts = tuple(part for part in stable if REAPER_NAME_PREDICATE.search(part))
    matched_tokens = tuple(
        sorted({match.group(0) for part in stable for match in REAPER_NAME_PREDICATE.finditer(part)})
    )
    stable_line = " ".join(stable)
    unresolved_env_wrapper = bool(
        requested and Path(requested[0]).name == "env" and stable == requested
    )
    exemption_hits = (
        ()
        if unresolved_env_wrapper
        else tuple(
            sorted({match.group(0) for match in REAPER_EXEMPTION_PREDICATE.finditer(stable_line)})
        )
    )
    return FleetReaperAssessment(
        requested_argv=requested,
        stable_argv=stable,
        matched_tokens=matched_tokens,
        matching_argv_parts=matching_parts,
        exemption_hits=exemption_hits,
    )


def assert_detached_argv_reaper_safe(argv: Sequence[str]) -> FleetReaperAssessment:
    """Return the assessment or refuse before any detached process exists."""

    assessment = assess_detached_argv(argv)
    if assessment.refused:
        raise FleetReaperLaunchRefusal(assessment)
    return assessment
