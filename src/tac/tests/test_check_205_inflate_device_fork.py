# SPDX-License-Identifier: MIT
"""Catalog #205 - check_inflate_py_uses_canonical_select_inflate_device tests.

A1 council Round 1 finding F1/F11 (CRITICAL, 2026-05-13). Inline
``device = "cuda" if torch.cuda.is_available() else "cpu"`` at top level
in ``submissions/*/inflate.py`` is the empirical mechanism producing the
+0.0335 CPU/CUDA score gap on the same archive bytes.

This test file exercises:

* live-repo regression guard (live count = 0)
* positive detection of bare top-level inline pattern
* positive detection inside non-canonical helper (wrong helper name)
* positive detection of ``torch.device("cuda" if ... else "cpu")`` wrapper
* negative acceptance inside canonical ``select_inflate_device`` helper
* negative acceptance when helper is correctly defined + honors
  ``PACT_INFLATE_DEVICE``
* negative on helper present without env-var policy (bug class - helper
  is canonical name but does not honor env contract)
* same-line ``# INLINE_DEVICE_FORK_OK:<reason>`` waiver acceptance
* placeholder ``<reason>`` literal rejection (no self-waiver)
* multiple violations in one file
* multiple files aggregated
* ``submissions/exact_current/`` is exempt (pinned upstream snapshot)
* ``experiments/results/`` is out-of-scope (DERIVED_OUTPUT)
* empty repo / no submissions/ dir is OK
* malformed Python tolerated (does not raise)
* strict mode raises on any violation
* non-strict mode returns list without raising
* unrelated assignments not flagged
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_inflate_py_uses_canonical_select_inflate_device,
)


# ---------------------------------------------------------------------------
# Canonical inline templates used by the synthetic-fixture tests below.
# ---------------------------------------------------------------------------

CANONICAL_HELPER = '''
import os
import torch


def select_inflate_device() -> torch.device:
    policy = os.environ.get("PACT_INFLATE_DEVICE", "auto").strip().lower()
    if policy in {"mps", "metal"}:
        raise RuntimeError("PACT_INFLATE_DEVICE=mps is forbidden")
    if policy == "cpu":
        return torch.device("cpu")
    if policy == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PACT_INFLATE_DEVICE=cuda but CUDA unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inflate(src_bin, dst_raw):
    device = select_inflate_device()
    return device
'''

BARE_INLINE_TOP = '''
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
'''

BARE_INLINE_STR = '''
import torch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device
'''

WRONG_HELPER_NAME = '''
import os
import torch


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inflate(src_bin, dst_raw):
    device = _select_device()
    return device
'''

CANONICAL_HELPER_NO_ENV = '''
import torch


def select_inflate_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inflate(src_bin, dst_raw):
    device = select_inflate_device()
    return device
'''


def _make_submission(repo_root: Path, lane: str, body: str) -> Path:
    sub = repo_root / "submissions" / lane
    sub.mkdir(parents=True, exist_ok=True)
    target = sub / "inflate.py"
    target.write_text(body, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Live-repo regression guard.
# ---------------------------------------------------------------------------


def test_check_205_live_repo_clean() -> None:
    """Live repo MUST have zero violations (Strict-flip atomicity rule)."""

    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=Path.cwd(), strict=False
    )
    assert violations == [], (
        f"Expected 0 violations on live repo, got {len(violations)}:\n  "
        + "\n  ".join(violations[:5])
    )


# ---------------------------------------------------------------------------
# Positive: detect bare inline patterns.
# ---------------------------------------------------------------------------


def test_check_205_detects_bare_torch_device_inline(tmp_path: Path) -> None:
    _make_submission(tmp_path, "lane_a", BARE_INLINE_TOP)
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1
    assert "lane_a/inflate.py" in violations[0]
    assert "device-fork OUTSIDE" in violations[0]


def test_check_205_detects_bare_string_inline(tmp_path: Path) -> None:
    _make_submission(tmp_path, "lane_b", BARE_INLINE_STR)
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1
    assert "lane_b/inflate.py" in violations[0]


def test_check_205_detects_wrong_helper_name(tmp_path: Path) -> None:
    """Helper named `_select_device` (not `select_inflate_device`) is wrong."""

    _make_submission(tmp_path, "lane_c", WRONG_HELPER_NAME)
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) >= 1
    assert "lane_c/inflate.py" in violations[0]


def test_check_205_detects_helper_without_env_var(tmp_path: Path) -> None:
    """Canonical helper name but no PACT_INFLATE_DEVICE reference is a bug.

    The helper must honor the env-var policy so the operator can pin the
    inflate device explicitly. A helper that ONLY contains the inline
    fallback is dead engineering - it looks canonical but does not
    actually give the operator any control.
    """

    _make_submission(tmp_path, "lane_d", CANONICAL_HELPER_NO_ENV)
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1
    assert "PACT_INFLATE_DEVICE" in violations[0]


# ---------------------------------------------------------------------------
# Negative: canonical helper accepted.
# ---------------------------------------------------------------------------


def test_check_205_accepts_canonical_helper(tmp_path: Path) -> None:
    _make_submission(tmp_path, "lane_canonical", CANONICAL_HELPER)
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_check_205_accepts_canonical_helper_with_extra_top_level_code(
    tmp_path: Path,
) -> None:
    body = (
        CANONICAL_HELPER
        + "\n\nCAMERA_HW = (874, 1164)\n\n"
        "def parse_archive(buf):\n    return buf\n"
    )
    _make_submission(tmp_path, "lane_extra", body)
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Same-line waiver.
# ---------------------------------------------------------------------------


def test_check_205_same_line_waiver_accepts(tmp_path: Path) -> None:
    body = '''
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # INLINE_DEVICE_FORK_OK:legacy-research-only inflate path
    return device
'''
    _make_submission(tmp_path, "lane_waiver", body)
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_check_205_placeholder_reason_rejected(tmp_path: Path) -> None:
    """Placeholder ``<reason>`` literal MUST NOT silence the gate."""

    body = '''
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # INLINE_DEVICE_FORK_OK:<reason>
    return device
'''
    _make_submission(tmp_path, "lane_placeholder", body)
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1


def test_check_205_empty_waiver_reason_rejected(tmp_path: Path) -> None:
    """Empty waiver tail MUST NOT silence the gate."""

    body = '''
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # INLINE_DEVICE_FORK_OK:
    return device
'''
    _make_submission(tmp_path, "lane_empty_waiver", body)
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Multiple violations.
# ---------------------------------------------------------------------------


def test_check_205_multiple_violations_in_one_file(tmp_path: Path) -> None:
    body = '''
import torch


def main():
    device_a = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_b = "cuda" if torch.cuda.is_available() else "cpu"
    return device_a, device_b
'''
    _make_submission(tmp_path, "lane_multi", body)
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 2


def test_check_205_aggregates_across_files(tmp_path: Path) -> None:
    _make_submission(tmp_path, "lane_a", BARE_INLINE_TOP)
    _make_submission(tmp_path, "lane_b", BARE_INLINE_STR)
    _make_submission(tmp_path, "lane_canonical", CANONICAL_HELPER)
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 2
    files = {v.split(":", 1)[0] for v in violations}
    assert any("lane_a" in f for f in files)
    assert any("lane_b" in f for f in files)
    assert not any("lane_canonical" in f for f in files)


# ---------------------------------------------------------------------------
# Scope: submissions/exact_current/ exempt; experiments/results/ out-of-scope.
# ---------------------------------------------------------------------------


def test_check_205_exact_current_is_exempt(tmp_path: Path) -> None:
    """``submissions/exact_current/`` is the pinned upstream snapshot."""

    _make_submission(tmp_path, "exact_current", BARE_INLINE_TOP)
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_check_205_experiments_results_out_of_scope(tmp_path: Path) -> None:
    """``experiments/results/`` is DERIVED_OUTPUT (Catalog #113)."""

    sub = tmp_path / "experiments" / "results" / "old_dispatch" / "submission_dir"
    sub.mkdir(parents=True)
    (sub / "inflate.py").write_text(BARE_INLINE_TOP, encoding="utf-8")
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------------


def test_check_205_no_submissions_dir_is_ok(tmp_path: Path) -> None:
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_check_205_empty_inflate_py_is_ok(tmp_path: Path) -> None:
    _make_submission(tmp_path, "lane_empty", "")
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_check_205_malformed_python_tolerated(tmp_path: Path) -> None:
    _make_submission(tmp_path, "lane_broken", "def broken( :\n  pass\n")
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_check_205_unrelated_assignment_not_flagged(tmp_path: Path) -> None:
    """Plain string assignments without the ternary MUST NOT trigger."""

    body = '''
import torch


def main():
    device = "cpu"
    other = torch.device("cuda")
    return device
'''
    _make_submission(tmp_path, "lane_unrelated", body)
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_check_205_helper_internal_fallback_not_flagged(tmp_path: Path) -> None:
    """Inline pattern INSIDE ``select_inflate_device`` body is canonical."""

    body = '''
import os
import torch


def select_inflate_device():
    policy = os.environ.get("PACT_INFLATE_DEVICE", "auto").strip().lower()
    if policy == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inflate(src_bin, dst_raw):
    device = select_inflate_device()
    return device
'''
    _make_submission(tmp_path, "lane_helper_internal", body)
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Strict-mode behavior.
# ---------------------------------------------------------------------------


def test_check_205_strict_raises_preflight_error(tmp_path: Path) -> None:
    _make_submission(tmp_path, "lane_violation", BARE_INLINE_TOP)
    with pytest.raises(PreflightError) as exc:
        check_inflate_py_uses_canonical_select_inflate_device(
            repo_root=tmp_path, strict=True
        )
    assert "F1/F11" in str(exc.value)
    assert "0.0335" in str(exc.value)


def test_check_205_strict_silent_when_clean(tmp_path: Path) -> None:
    _make_submission(tmp_path, "lane_canonical", CANONICAL_HELPER)
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=True
    )
    assert violations == []


def test_check_205_returns_list_when_not_strict(tmp_path: Path) -> None:
    _make_submission(tmp_path, "lane_violation", BARE_INLINE_TOP)
    violations = check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False
    )
    assert isinstance(violations, list)
    assert len(violations) == 1


def test_check_205_verbose_emits_summary(tmp_path: Path, capsys) -> None:
    _make_submission(tmp_path, "lane_canonical", CANONICAL_HELPER)
    check_inflate_py_uses_canonical_select_inflate_device(
        repo_root=tmp_path, strict=False, verbose=True
    )
    captured = capsys.readouterr().out
    assert "[inflate-device-fork]" in captured
