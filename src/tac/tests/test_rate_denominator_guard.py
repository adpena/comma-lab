# SPDX-License-Identifier: MIT
"""Tests for the rate-denominator cleanliness guard (task #812, Catalog #407).

The contest rate DENOMINATOR is dynamic: ``upstream/evaluate.py:64`` computes it
as ``sum(f.stat().st_size for f in uncompressed_dir.rglob('*') if f.is_file())``.
``rglob('*')`` COUNTS DOTFILES, so a stray macOS ``._0.mkv`` (AppleDouble) or
``.DS_Store`` silently inflates the denominator and corrupts every score. These
tests cover the two-landing protection:

* Landing 1 — ``tac.contest_score`` helper + fail-closed guard at ``rate_term``.
* Landing 2 — warn-only ``tac.preflight.check_upstream_videos_dir_clean``.

Every fixture uses a tmp tree; the REAL ``upstream/videos/`` is never touched.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.contest_score import (
    UNCOMPRESSED_SIZE_BYTES,
    RateDenominatorMismatchError,
    RateDenominatorVerdict,
    _assert_default_denominator_clean_cached,
    _reset_denominator_cache,
    assert_upstream_videos_clean,
    compute_contest_score,
    expected_video_names,
    rate_term,
    verify_upstream_videos_clean,
)
from tac.preflight import check_upstream_videos_dir_clean


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _make_videos_dir(tmp_path: Path, *, mkv_bytes: int = 1000, extra: dict | None = None) -> Path:
    """A bare ``videos/`` dir with a clean ``0.mkv`` plus optional extra files."""
    d = tmp_path / "videos"
    d.mkdir()
    (d / "0.mkv").write_bytes(b"x" * mkv_bytes)
    for name, size in (extra or {}).items():
        (d / name).write_bytes(b"y" * size)
    return d


def _make_repo_root(tmp_path: Path, *, mkv_bytes: int = 1000, extra: dict | None = None) -> Path:
    """A minimal repo root: ``upstream/videos/0.mkv`` + names file."""
    (tmp_path / "upstream").mkdir()
    (tmp_path / "upstream" / "public_test_video_names.txt").write_text("0.mkv\n")
    vids = tmp_path / "upstream" / "videos"
    vids.mkdir()
    (vids / "0.mkv").write_bytes(b"x" * mkv_bytes)
    for name, size in (extra or {}).items():
        (vids / name).write_bytes(b"y" * size)
    return tmp_path


@pytest.fixture(autouse=True)
def _clear_cache():
    """The rate_term default-denominator verdict is cached per-process; reset it
    around every test so cross-test injection never leaks."""
    _reset_denominator_cache()
    yield
    _reset_denominator_cache()


# --------------------------------------------------------------------------- #
# Landing 1 — verify_upstream_videos_clean (measurement, never raises)
# --------------------------------------------------------------------------- #
def test_verify_clean_tree_reports_clean(tmp_path):
    d = _make_videos_dir(tmp_path, mkv_bytes=1000)
    v = verify_upstream_videos_clean(d, expected_sum=1000, expected_names=("0.mkv",))
    assert isinstance(v, RateDenominatorVerdict)
    assert v.present and v.clean
    assert v.strays == () and v.missing == () and v.sum_matches
    assert v.dynamic_sum == 1000
    assert "CLEAN" in v.report


def test_verify_replicates_evaluate_line64_counts_dotfiles(tmp_path):
    # THE vulnerability: rglob('*') is_file counts a dotfile. This is the exact
    # arithmetic of upstream/evaluate.py:64, so our helper must count it too.
    d = _make_videos_dir(tmp_path, mkv_bytes=1000, extra={"._0.mkv": 50})
    v = verify_upstream_videos_clean(d, expected_sum=1000, expected_names=("0.mkv",))
    assert v.dynamic_sum == 1050  # 1000 + 50 stray → denominator inflated
    assert not v.clean
    assert "._0.mkv" in v.strays


def test_verify_appledouble_is_stray(tmp_path):
    d = _make_videos_dir(tmp_path, extra={"._0.mkv": 50})
    v = verify_upstream_videos_clean(d, expected_sum=1000, expected_names=("0.mkv",))
    assert v.strays == ("._0.mkv",)
    assert not v.clean


def test_verify_dsstore_is_stray(tmp_path):
    d = _make_videos_dir(tmp_path, extra={".DS_Store": 30})
    v = verify_upstream_videos_clean(d, expected_sum=1000, expected_names=("0.mkv",))
    assert v.strays == (".DS_Store",)
    assert not v.clean
    assert ".DS_Store" in v.report


def test_verify_unexpected_named_file_is_stray(tmp_path):
    d = _make_videos_dir(tmp_path, extra={"1.mkv": 20})
    v = verify_upstream_videos_clean(d, expected_sum=1000, expected_names=("0.mkv",))
    assert "1.mkv" in v.strays
    assert not v.clean


def test_verify_missing_expected_file(tmp_path):
    d = tmp_path / "videos"
    d.mkdir()  # empty — 0.mkv missing
    v = verify_upstream_videos_clean(d, expected_sum=1000, expected_names=("0.mkv",))
    assert v.missing == ("0.mkv",)
    assert not v.clean
    assert "MISSING" in v.report


def test_verify_wrong_size_sum_mismatch(tmp_path):
    d = _make_videos_dir(tmp_path, mkv_bytes=999)  # expected 1000
    v = verify_upstream_videos_clean(d, expected_sum=1000, expected_names=("0.mkv",))
    assert not v.sum_matches
    assert not v.clean
    assert "BYTE-SUM" in v.report and "999" in v.report


def test_verify_absent_dir_unverifiable_not_a_violation(tmp_path):
    v = verify_upstream_videos_clean(tmp_path / "nope", expected_sum=1000)
    assert not v.present
    assert v.clean  # absent → unverifiable → constant stands; never a violation
    assert "UNVERIFIABLE" in v.report


def test_verify_nested_subdir_stray_counted(tmp_path):
    # rglob is recursive → a stray in a subdir also inflates the denominator.
    d = _make_videos_dir(tmp_path, mkv_bytes=1000)
    sub = d / "sub"
    sub.mkdir()
    (sub / ".DS_Store").write_bytes(b"z" * 10)
    v = verify_upstream_videos_clean(d, expected_sum=1000, expected_names=("0.mkv",))
    assert "sub/.DS_Store" in v.strays
    assert v.dynamic_sum == 1010


# --------------------------------------------------------------------------- #
# Landing 1 — assert_upstream_videos_clean (fail-closed)
# --------------------------------------------------------------------------- #
def test_assert_clean_returns_verdict(tmp_path):
    d = _make_videos_dir(tmp_path, mkv_bytes=1000)
    v = assert_upstream_videos_clean(d, expected_sum=1000, expected_names=("0.mkv",))
    assert v.clean


def test_assert_dirty_raises_naming_stray(tmp_path):
    d = _make_videos_dir(tmp_path, extra={"._0.mkv": 50})
    with pytest.raises(RateDenominatorMismatchError, match=r"\._0\.mkv"):
        assert_upstream_videos_clean(d, expected_sum=1000, expected_names=("0.mkv",))


def test_assert_absent_does_not_raise(tmp_path):
    v = assert_upstream_videos_clean(tmp_path / "nope", expected_sum=1000)
    assert not v.present  # no raise on unverifiable tree


def test_assert_never_deletes_the_stray(tmp_path):
    d = _make_videos_dir(tmp_path, extra={".DS_Store": 30})
    with pytest.raises(RateDenominatorMismatchError):
        assert_upstream_videos_clean(d, expected_sum=1000, expected_names=("0.mkv",))
    # upstream/ is immutable — the guard must NEVER remove the file.
    assert (d / ".DS_Store").exists()


# --------------------------------------------------------------------------- #
# Landing 1 — rate_term wire-in (guard fires only on the default constant)
# --------------------------------------------------------------------------- #
def test_rate_term_clean_real_tree_no_raise():
    # The real upstream/videos/ tree is clean; rate_term must work normally.
    _reset_denominator_cache()
    assert rate_term(190952) == pytest.approx(25 * 190952 / UNCOMPRESSED_SIZE_BYTES)


def test_rate_term_fail_closed_on_dirty_cached_verdict():
    import tac.contest_score as cs

    cs._DEFAULT_DENOMINATOR_VERDICT = RateDenominatorVerdict(
        present=True, clean=False, videos_dir="x", dynamic_sum=99,
        expected_sum=UNCOMPRESSED_SIZE_BYTES, expected_files=("0.mkv",),
        actual_files=("0.mkv", ".DS_Store"), strays=(".DS_Store",),
        missing=(), sum_matches=False, report="DIRTY: stray .DS_Store",
    )
    with pytest.raises(RateDenominatorMismatchError):
        rate_term(190952)  # default constant path → guard fires
    with pytest.raises(RateDenominatorMismatchError):
        compute_contest_score(0.004, 0.0016, 190952)  # routes through rate_term


def test_rate_term_explicit_nondefault_denominator_bypasses_guard():
    import tac.contest_score as cs

    # Inject a dirty verdict; an explicit non-canonical denominator is a
    # deliberate hypothetical NOT claiming the real contest denominator, so the
    # guard must be skipped and the arithmetic proceed.
    cs._DEFAULT_DENOMINATOR_VERDICT = RateDenominatorVerdict(
        present=True, clean=False, videos_dir="x", dynamic_sum=99,
        expected_sum=UNCOMPRESSED_SIZE_BYTES, expected_files=("0.mkv",),
        actual_files=("0.mkv",), strays=(".DS_Store",), missing=(),
        sum_matches=False, report="DIRTY",
    )
    assert rate_term(100, uncompressed_size=1000) == pytest.approx(2.5)


def test_cached_guard_absent_tree_is_noop(monkeypatch):
    import tac.contest_score as cs

    monkeypatch.setattr(cs, "UPSTREAM_VIDEOS_DIR", Path("/definitely/not/here/videos"))
    _reset_denominator_cache()
    _assert_default_denominator_clean_cached()  # must not raise
    assert cs._DEFAULT_DENOMINATOR_VERDICT is not None
    assert not cs._DEFAULT_DENOMINATOR_VERDICT.present


# --------------------------------------------------------------------------- #
# expected_video_names
# --------------------------------------------------------------------------- #
def test_expected_video_names_returns_names():
    names = expected_video_names()
    assert isinstance(names, tuple)
    assert "0.mkv" in names  # the canonical single-video contest set


# --------------------------------------------------------------------------- #
# Landing 2 — check_upstream_videos_dir_clean (warn-only preflight)
# --------------------------------------------------------------------------- #
def test_preflight_clean_fixture_zero_violations(tmp_path):
    root = _make_repo_root(tmp_path)
    assert check_upstream_videos_dir_clean(repo_root=root) == []


def test_preflight_appledouble_refused_and_named(tmp_path):
    root = _make_repo_root(tmp_path, extra={"._0.mkv": 50})
    vio = check_upstream_videos_dir_clean(repo_root=root)
    assert len(vio) == 1
    assert "._0.mkv" in vio[0]
    assert "STRAY" in vio[0]


def test_preflight_dsstore_refused(tmp_path):
    root = _make_repo_root(tmp_path, extra={".DS_Store": 30})
    vio = check_upstream_videos_dir_clean(repo_root=root)
    assert len(vio) == 1
    assert ".DS_Store" in vio[0]


def test_preflight_missing_payload_refused(tmp_path):
    (tmp_path / "upstream").mkdir()
    (tmp_path / "upstream" / "public_test_video_names.txt").write_text("0.mkv\n")
    (tmp_path / "upstream" / "videos").mkdir()  # empty
    vio = check_upstream_videos_dir_clean(repo_root=tmp_path)
    assert len(vio) == 1
    assert "MISSING" in vio[0]


def test_preflight_strict_raises_on_dirty(tmp_path):
    from tac.preflight import PreflightError

    root = _make_repo_root(tmp_path, extra={".DS_Store": 30})
    with pytest.raises(PreflightError, match="rate-denominator contamination"):
        check_upstream_videos_dir_clean(repo_root=root, strict=True)


def test_preflight_absent_videos_dir_not_a_violation(tmp_path):
    (tmp_path / "upstream").mkdir()  # no videos/ dir at all
    assert check_upstream_videos_dir_clean(repo_root=tmp_path) == []


def test_preflight_names_file_fallback_when_absent(tmp_path):
    # No names file → the gate falls back to the canonical ("0.mkv",) inventory.
    (tmp_path / "upstream").mkdir()
    vids = tmp_path / "upstream" / "videos"
    vids.mkdir()
    (vids / "0.mkv").write_bytes(b"x" * 100)
    (vids / ".DS_Store").write_bytes(b"z" * 5)
    vio = check_upstream_videos_dir_clean(repo_root=tmp_path)
    assert any(".DS_Store" in v for v in vio)


def test_preflight_real_repo_live_count_zero():
    # The live upstream/videos/ tree is clean at landing (documented live count 0
    # → the warn-only wire-in is honest, and a future strict-flip is unblocked).
    assert check_upstream_videos_dir_clean() == []
