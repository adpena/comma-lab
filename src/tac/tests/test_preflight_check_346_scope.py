# SPDX-License-Identifier: MIT
"""Catalog #346 gate scope controls (arm cr1, 2026-08-25).

`check_council_dispatch_roster_complete_per_canonical_helper` selects memos by
FILENAME GLOB over `.omx/research/`. Two scope defects were burning the live
violation count without any real under-rostering:

  (1) DAG-FEED COMPANIONS. `council_*_DAG_FEED_*.md` documents are triality
      trajectory records OF a deliberation, not the deliberation dispatch. They
      decline to re-enumerate the seated roster and name the parent council
      memo instead. Demanding a dispatch roster from one is a scope defect.
  (2) ANACHRONISTIC DEMAND. The roster is APPEND-ONLY (Catalog #110/#113), so a
      memo was being required to seat members added AFTER it was convened.

Every test below is paired: the exclusion must hold AND the gate must still
fire on the equivalent in-scope case. A scope fix that stops catching real
under-rostering is a laundered count, not a cure.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac import preflight
from tac.preflight import (
    PreflightError,
    check_council_dispatch_roster_complete_per_canonical_helper,
)


# A T2 memo seating only 8 of the 14 mandatory inner-council seats — genuinely
# under-rostered under any reading, so it is a sound positive control.
_UNDER_ROSTERED_FRONTMATTER = """---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich,
  Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED
---

# body
"""

# A T3 memo whose full inner council IS seated, on an hnerv/nerv topic. The only
# seats it can be missing are the NeRV grand seats appended 2026-06-01.
_ERA_SENSITIVE_FRONTMATTER = """---
council_tier: T3
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich,
  Contrarian, Assumption-Adversary, Quantizr, Hotz, Selfcomp, MacKay, Balle,
  PR95Author, Boyd, Tao, Filler, Mallat, vdOord, Carmack, Hassabis, Hinton,
  Karpathy, Schmidhuber, JackFromSkunkworks, Atick, Redlich, Rao, Ballard,
  Tishby, Zaslavsky, Wyner, TimeTraveler, TimeTravelerProtege, Rudin_Grand,
  Daubechies_Grand]
topic_tokens: [hnerv, nerv, rnerv, carrier_architecture]
council_quorum_met: true
council_verdict: PROCEED
---

# body
"""


def _make_repo(tmp_path: Path, memos: dict[str, str]) -> Path:
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    for name, text in memos.items():
        (research / name).write_text(text, encoding="utf-8")
    return tmp_path


def _run(root: Path) -> list[str]:
    return check_council_dispatch_roster_complete_per_canonical_helper(
        strict=False, verbose=False, repo_root=root,
    )


class TestDagFeedCompanionScope:
    def test_positive_control_normal_council_memo_still_fires(self, tmp_path: Path) -> None:
        """The SAME content under a normal council_ filename must be caught."""
        root = _make_repo(
            tmp_path, {"council_widget_bridge_20260728.md": _UNDER_ROSTERED_FRONTMATTER},
        )
        violations = _run(root)
        assert len(violations) == 1
        assert "council_widget_bridge_20260728.md" in violations[0]

    def test_dag_feed_companion_is_out_of_scope(self, tmp_path: Path) -> None:
        """Byte-identical content under a DAG-FEED filename is a companion."""
        root = _make_repo(
            tmp_path,
            {"council_widget_bridge_DAG_FEED_20260728.md": _UNDER_ROSTERED_FRONTMATTER},
        )
        assert _run(root) == []

    def test_companion_exclusion_does_not_hide_its_parent(self, tmp_path: Path) -> None:
        """Coverage preservation: the parent deliberation memo is still caught."""
        root = _make_repo(
            tmp_path,
            {
                "council_widget_bridge_DAG_FEED_20260728.md": _UNDER_ROSTERED_FRONTMATTER,
                "council_widget_bridge_20260728.md": _UNDER_ROSTERED_FRONTMATTER,
            },
        )
        violations = _run(root)
        assert len(violations) == 1
        assert "DAG_FEED" not in violations[0]

    def test_live_repo_dag_feed_has_an_in_scope_parent(self) -> None:
        """The 2026-08-25 live companion names a parent that IS validated."""
        research = preflight.REPO_ROOT / ".omx" / "research"
        companion = research / "council_gc5_micro_macro_bridge_DAG_FEED_20260728.md"
        parent = research / "council_gc5_schmidhuber_micro_macro_bridge_20260728.md"
        if not companion.exists():
            pytest.skip("live companion memo absent")
        assert parent.exists(), "companion excluded but parent memo missing"
        # The companion cites its parent by council-anchor id (the filename
        # stem), not by filename-with-extension.
        assert parent.stem in companion.read_text(encoding="utf-8")


class TestSeatAvailabilityEraScope:
    def test_memo_predating_seat_addition_is_not_anachronistically_charged(
        self, tmp_path: Path,
    ) -> None:
        """2026-05-20 memo cannot owe the NeRV seats added 2026-06-01."""
        root = _make_repo(
            tmp_path, {"council_dwt_bind_20260520.md": _ERA_SENSITIVE_FRONTMATTER},
        )
        assert _run(root) == []

    def test_positive_control_same_memo_after_seat_addition_fires(
        self, tmp_path: Path,
    ) -> None:
        """Byte-identical content dated after 2026-06-01 DOES owe those seats."""
        root = _make_repo(
            tmp_path, {"council_dwt_bind_20260615.md": _ERA_SENSITIVE_FRONTMATTER},
        )
        violations = _run(root)
        assert len(violations) == 1
        assert "HaoChen_NeRV" in violations[0] or "Shrivastava_INR" in violations[0]

    def test_era_scope_does_not_excuse_inner_council_omission(
        self, tmp_path: Path,
    ) -> None:
        """NEGATIVE control: no memo date excuses an inner-council seat.

        All 14 inner seats landed with the roster on 2026-05-19, so the era
        filter can never reach them.
        """
        root = _make_repo(
            tmp_path, {"council_early_20260519.md": _UNDER_ROSTERED_FRONTMATTER},
        )
        violations = _run(root)
        assert len(violations) == 1
        assert "Quantizr" in violations[0]


class TestWaiverContract:
    def test_substantive_waiver_accepted(self, tmp_path: Path) -> None:
        text = _UNDER_ROSTERED_FRONTMATTER + (
            "\n<!-- # COUNCIL_ROSTER_INCOMPLETE_OK:under_rostered_convocation_"
            "acknowledged_append_only_per_catalog_110_113 -->\n"
        )
        root = _make_repo(tmp_path, {"council_widget_20260728.md": text})
        assert _run(root) == []

    def test_placeholder_waiver_rejected(self, tmp_path: Path) -> None:
        text = _UNDER_ROSTERED_FRONTMATTER + (
            "\n<!-- # COUNCIL_ROSTER_INCOMPLETE_OK:<rationale> -->\n"
        )
        root = _make_repo(tmp_path, {"council_widget_20260728.md": text})
        assert len(_run(root)) == 1

    def test_strict_raises(self, tmp_path: Path) -> None:
        root = _make_repo(
            tmp_path, {"council_widget_20260728.md": _UNDER_ROSTERED_FRONTMATTER},
        )
        with pytest.raises(PreflightError):
            check_council_dispatch_roster_complete_per_canonical_helper(
                strict=True, verbose=False, repo_root=root,
            )
