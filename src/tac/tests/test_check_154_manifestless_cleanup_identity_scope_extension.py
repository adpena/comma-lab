# SPDX-License-Identifier: MIT
"""Adversarial tests for the Catalog #154 manifest-less identity extension."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _check_154_manifestless_cleanup_identity,
    check_experiments_results_gc_helper_is_canonical,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
_EXACT_KEYS = ("path", "bytes", "sha256")


def _write_python(repo: Path, relative_path: str, source: str) -> Path:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    return path


def _manifestless_source(
    branch: str,
    *,
    final_keys: tuple[str, ...] = _EXACT_KEYS,
    precleanup_keys: tuple[str, ...] = _EXACT_KEYS,
    preserved_files: str = "both",
    compare_receipt_sha: bool = True,
    compare_predecessor_exactly: bool = True,
) -> str:
    """Build a small producer with a structurally inspectable frozen validator."""

    final_values = {
        "path": ".omx/research/final.json",
        "bytes": 101,
        "sha256": "f" * 64,
    }
    precleanup_values = {
        "path": ".omx/research/final.json.precleanup.json",
        "bytes": 97,
        "sha256": "e" * 64,
    }
    final_mapping = {key: final_values[key] for key in final_keys}
    precleanup_mapping = {key: precleanup_values[key] for key in precleanup_keys}

    if preserved_files == "both":
        preserved_lines = """
    final_path = _validate_preserved_file(FROZEN_FINAL_RECEIPT, "final")
    precleanup_path = _validate_preserved_file(
        FROZEN_PRECLEANUP_RECEIPT, "precleanup"
    )
"""
    elif preserved_files == "final-only":
        preserved_lines = """
    final_path = _validate_preserved_file(FROZEN_FINAL_RECEIPT, "final")
    precleanup_path = (REPO / FROZEN_PRECLEANUP_RECEIPT["path"]).resolve()
"""
    elif preserved_files == "final-twice":
        # Token-copying is not custody: the predecessor itself is never validated.
        preserved_lines = """
    final_path = _validate_preserved_file(FROZEN_FINAL_RECEIPT, "final")
    precleanup_path = _validate_preserved_file(FROZEN_FINAL_RECEIPT, "lookalike")
"""
    else:  # pragma: no cover - fixture authoring guard
        raise AssertionError(f"unknown preserved_files mode: {preserved_files}")

    receipt_sha_clause = (
        'receipt_sha256 != FROZEN_FINAL_RECEIPT["sha256"]'
        if compare_receipt_sha
        else "receipt_sha256 != receipt_sha256"
    )
    predecessor_clause = (
        "dict(predecessor) != FROZEN_PRECLEANUP_RECEIPT" if compare_predecessor_exactly else "predecessor is None"
    )
    indented_branch = textwrap.indent(textwrap.dedent(branch).strip(), "    ")
    return (
        "from pathlib import Path\n\n"
        "REPO = Path('.')\n"
        f"FROZEN_FINAL_RECEIPT = {final_mapping!r}\n"
        f"FROZEN_PRECLEANUP_RECEIPT = {precleanup_mapping!r}\n\n"
        "def _validate_preserved_file(row, label):\n"
        "    return (REPO / row['path']).resolve()\n\n"
        "def _validate_cleanup_identity(receipt):\n"
        "    receipt_sha256 = receipt['sha256']\n"
        f"{preserved_lines.strip(chr(10))}\n"
        "    predecessor = receipt.get('precleanup_receipt')\n"
        "    if (\n"
        "        final_path != (REPO / FROZEN_FINAL_RECEIPT['path']).resolve()\n"
        "        or precleanup_path != (REPO / FROZEN_PRECLEANUP_RECEIPT['path']).resolve()\n"
        f"        or {receipt_sha_clause}\n"
        f"        or {predecessor_clause}\n"
        "    ):\n"
        "        raise RuntimeError('historical cleanup identity drifted')\n\n"
        "def consume(lifecycle, receipt):\n"
        f"{indented_branch}\n"
    )


def _scan_source(tmp_path: Path, source: str, relative_path: str = "tools/producer.py") -> list[str]:
    _write_python(tmp_path, relative_path, source)
    return _check_154_manifestless_cleanup_identity(tmp_path)


def _assert_one_identity_violation(violations: list[str], relative_path: str = "tools/producer.py") -> None:
    assert len(violations) == 1
    assert relative_path in violations[0]
    assert "manifest-less cleanup compatibility" in violations[0]
    assert "Catalog #154 scope extension" in violations[0]


@pytest.mark.parametrize(
    "branch",
    [
        pytest.param(
            """
            if "cleanup_manifest" not in lifecycle:
                _validate_cleanup_identity(receipt)
            """,
            id="negative-membership-body-is-absence",
        ),
        pytest.param(
            """
            if "cleanup_manifest" in lifecycle:
                return "manifest-present"
            else:
                _validate_cleanup_identity(receipt)
            """,
            id="positive-membership-else-is-absence",
        ),
        pytest.param(
            """
            if lifecycle.get("cleanup_manifest") is None:
                _validate_cleanup_identity(receipt)
            """,
            id="get-is-none-body-is-absence",
        ),
        pytest.param(
            """
            if lifecycle.get("cleanup_manifest") is not None:
                return "manifest-present"
            else:
                _validate_cleanup_identity(receipt)
            """,
            id="get-is-not-none-else-is-absence",
        ),
    ],
)
def test_exact_final_and_precleanup_identity_accepts_each_absence_form(
    tmp_path: Path,
    branch: str,
) -> None:
    source = _manifestless_source(branch)
    assert _scan_source(tmp_path, source) == []


@pytest.mark.parametrize(
    ("mapping", "missing_key"),
    [
        pytest.param("final", "path", id="final-missing-path"),
        pytest.param("final", "bytes", id="final-missing-bytes"),
        pytest.param("final", "sha256", id="final-missing-sha256"),
        pytest.param("precleanup", "path", id="precleanup-missing-path"),
        pytest.param("precleanup", "bytes", id="precleanup-missing-bytes"),
        pytest.param("precleanup", "sha256", id="precleanup-missing-sha256"),
    ],
)
def test_frozen_mapping_missing_any_exact_identity_key_is_rejected(
    tmp_path: Path,
    mapping: str,
    missing_key: str,
) -> None:
    final_keys = tuple(key for key in _EXACT_KEYS if not (mapping == "final" and key == missing_key))
    precleanup_keys = tuple(key for key in _EXACT_KEYS if not (mapping == "precleanup" and key == missing_key))
    source = _manifestless_source(
        """
        if "cleanup_manifest" not in lifecycle:
            _validate_cleanup_identity(receipt)
        """,
        final_keys=final_keys,
        precleanup_keys=precleanup_keys,
    )
    _assert_one_identity_violation(_scan_source(tmp_path, source))


def test_direct_fail_closed_raise_needs_no_legacy_identity_validator(tmp_path: Path) -> None:
    source = """
    def consume(lifecycle):
        if "cleanup_manifest" not in lifecycle:
            raise RuntimeError("manifest-less cleanup is unsupported")
    """
    assert _scan_source(tmp_path, source) == []


def test_only_one_preserved_file_validation_is_rejected(tmp_path: Path) -> None:
    source = _manifestless_source(
        """
        if "cleanup_manifest" not in lifecycle:
            _validate_cleanup_identity(receipt)
        """,
        preserved_files="final-only",
    )
    _assert_one_identity_violation(_scan_source(tmp_path, source))


def test_missing_receipt_sha_comparison_is_rejected(tmp_path: Path) -> None:
    source = _manifestless_source(
        """
        if lifecycle.get("cleanup_manifest") is None:
            _validate_cleanup_identity(receipt)
        """,
        compare_receipt_sha=False,
    )
    _assert_one_identity_violation(_scan_source(tmp_path, source))


def test_missing_exact_predecessor_comparison_is_rejected(tmp_path: Path) -> None:
    source = _manifestless_source(
        """
        if lifecycle.get("cleanup_manifest") is None:
            _validate_cleanup_identity(receipt)
        """,
        compare_predecessor_exactly=False,
    )
    _assert_one_identity_violation(_scan_source(tmp_path, source))


def test_manifestless_branch_not_routed_to_validator_is_rejected(tmp_path: Path) -> None:
    source = _manifestless_source(
        """
        if "cleanup_manifest" not in lifecycle:
            return "legacy-shape-accepted-without-validation"
        """
    )
    _assert_one_identity_violation(_scan_source(tmp_path, source))


def test_copied_validation_shape_cannot_substitute_final_for_precleanup(tmp_path: Path) -> None:
    source = _manifestless_source(
        """
        if "cleanup_manifest" not in lifecycle:
            _validate_cleanup_identity(receipt)
        """,
        preserved_files="final-twice",
    )
    _assert_one_identity_violation(_scan_source(tmp_path, source))


def test_substantive_same_line_waiver_accepts_deliberate_compatibility(tmp_path: Path) -> None:
    source = """
    def consume(lifecycle):
        if "cleanup_manifest" not in lifecycle:  # MANIFESTLESS_CLEANUP_IDENTITY_OK:frozen fixture has external notarized custody
            return "operator-reviewed compatibility"
    """
    assert _scan_source(tmp_path, source) == []


def test_placeholder_same_line_waiver_does_not_self_exempt(tmp_path: Path) -> None:
    source = """
    def consume(lifecycle):
        if "cleanup_manifest" not in lifecycle:  # MANIFESTLESS_CLEANUP_IDENTITY_OK:<rationale>
            return "uncustodied compatibility"
    """
    _assert_one_identity_violation(_scan_source(tmp_path, source))


def test_cleanup_manifest_producer_syntax_error_is_fail_closed(tmp_path: Path) -> None:
    violations = _scan_source(
        tmp_path,
        """
        cleanup_manifest = None
        def broken(:
            pass
        """,
    )
    assert len(violations) == 1
    assert "tools/producer.py" in violations[0]
    assert "not parseable" in violations[0]


@pytest.mark.parametrize(
    "layout",
    [
        pytest.param("tests-directory", id="tests-directory"),
        pytest.param("test-filename", id="test-filename"),
        pytest.param("self-exempt-files", id="canonical-self-exempt-files"),
    ],
)
def test_tests_and_canonical_self_files_are_exempt(tmp_path: Path, layout: str) -> None:
    bad_source = """
    def consume(lifecycle):
        if "cleanup_manifest" not in lifecycle:
            return "fixture-only legacy shape"
    """
    if layout == "tests-directory":
        _write_python(tmp_path, "tools/tests/cleanup_fixture.py", bad_source)
    elif layout == "test-filename":
        _write_python(tmp_path, "tools/test_cleanup_fixture.py", bad_source)
    else:
        _write_python(tmp_path, "tools/gc_experiments_results.py", bad_source)
        _write_python(tmp_path, "src/tac/preflight.py", bad_source)
    assert _check_154_manifestless_cleanup_identity(tmp_path) == []


def test_absent_scan_directories_are_clean(tmp_path: Path) -> None:
    assert _check_154_manifestless_cleanup_identity(tmp_path) == []
    assert (
        check_experiments_results_gc_helper_is_canonical(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )
        == []
    )


def test_strict_wrapper_raises_for_manifestless_identity_gap(tmp_path: Path) -> None:
    _write_python(
        tmp_path,
        "scripts/legacy_cleanup.py",
        """
        def consume(lifecycle):
            if lifecycle.get("cleanup_manifest") is None:
                return "uncustodied legacy cleanup"
        """,
    )
    with pytest.raises(
        PreflightError,
        match="manifest-less historical identity gaps",
    ):
        check_experiments_results_gc_helper_is_canonical(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )


def test_check_154_manifestless_extension_live_count_zero() -> None:
    assert _check_154_manifestless_cleanup_identity(REPO_ROOT) == []
    assert (
        check_experiments_results_gc_helper_is_canonical(
            repo_root=REPO_ROOT,
            strict=True,
            verbose=False,
        )
        == []
    )
