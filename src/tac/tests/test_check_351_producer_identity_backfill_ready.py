# SPDX-License-Identifier: MIT
"""Adversarial coverage for the GB1 Catalog #351 backfill guard."""

from __future__ import annotations

import inspect
from pathlib import Path
from textwrap import dedent

import pytest

from tac import preflight

_EXACT_HELPER = dedent(
    """
    def _canonical_producer_reference(value, *, canonical_path):
        candidate = Path(value)
        canonical_relative = Path(canonical_path)
        if (
            canonical_relative.is_absolute()
            or not canonical_relative.parts
            or ".." in canonical_relative.parts
            or ".." in candidate.parts
        ):
            raise ValueError("canonical producer paths must be non-empty repo-relative paths without parent traversal")
        requested = candidate.absolute() if candidate.is_absolute() else (REPO_ROOT / candidate).absolute()
        expected_requested = (REPO_ROOT / canonical_relative).absolute()
        if requested != expected_requested:
            raise ValueError("canonical producer alias refused")
        current = REPO_ROOT.absolute()
        root_lstat = current.lstat()
        if stat.S_ISLNK(root_lstat.st_mode):
            raise ValueError("canonical producer repo root must not be a symlink")
        if not stat.S_ISDIR(root_lstat.st_mode):
            raise ValueError("canonical producer repo root must be a directory")
        for part in canonical_relative.parts:
            current = current / part
            current_lstat = current.lstat()
            if stat.S_ISLNK(current_lstat.st_mode):
                raise ValueError("canonical producer path has a symlinked component")
            if current == expected_requested:
                if not stat.S_ISREG(current_lstat.st_mode):
                    raise ValueError("canonical producer non-file refused")
            elif not stat.S_ISDIR(current_lstat.st_mode):
                raise ValueError("canonical producer ancestor must be a directory")
        canonical_stat = expected_requested.stat()
        if canonical_stat.st_nlink != 1:
            raise ValueError("canonical producer hardlink refused")
        resolved = requested.resolve()
        expected = expected_requested.resolve()
        if resolved != expected:
            raise ValueError("canonical producer alias refused")
        return expected, canonical_path
    """
)

_SINGLE_BUILDER = dedent(
    """
    def build_contract(measurement_path=SOURCE_MEASUREMENT):
        input_path, source_label = _canonical_producer_reference(
            measurement_path,
            canonical_path=SOURCE_MEASUREMENT,
        )
        provenance = build_provenance_for_research_sidecar(sidecar_path=input_path)
        if provenance.source_sha256 != SOURCE_MEASUREMENT_SHA256:
            raise ValueError("hash drift")
        return dict(canonical_producers=(source_label,))
    """
)

_MULTI_BUILDER = dedent(
    """
    def build_contract(
        measurement_path=SOURCE_MEASUREMENT,
        frontier_path=SOURCE_FRONTIER,
    ):
        measurement_input, measurement_label = _canonical_producer_reference(
            measurement_path,
            canonical_path=SOURCE_MEASUREMENT,
        )
        frontier_input, frontier_label = _canonical_producer_reference(
            value=frontier_path,
            canonical_path=SOURCE_FRONTIER,
        )
        measurement_provenance = build_provenance_for_research_sidecar(
            sidecar_path=measurement_input
        )
        frontier_provenance = build_provenance_for_research_sidecar(
            sidecar_path=frontier_input
        )
        if measurement_provenance.source_sha256 != SOURCE_MEASUREMENT_SHA256:
            raise ValueError("measurement hash drift")
        if frontier_provenance.source_sha256 != SOURCE_FRONTIER_SHA256:
            raise ValueError("frontier hash drift")
        return dict(canonical_producers=(measurement_label, frontier_label))
    """
)


def _module_source(*, helper: str = _EXACT_HELPER, builder: str = _SINGLE_BUILDER) -> str:
    return (
        dedent(
            """
            import stat
            from pathlib import Path

            from tac.provenance.builders import build_provenance_for_research_sidecar

            REPO_ROOT = Path(__file__).resolve().parents[3]
            SOURCE_MEASUREMENT = ".omx/research/measurement.json"
            SOURCE_FRONTIER = ".omx/research/frontier.json"
            SOURCE_MEASUREMENT_SHA256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            SOURCE_FRONTIER_SHA256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            """
        )
        + helper
        + builder
    )


def _write_module(repo_root: Path, source: str, name: str = "candidate.py") -> Path:
    path = repo_root / "src/tac/canonical_equations" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _violations(repo_root: Path) -> list[str]:
    result = preflight._check_351_canonical_producer_identity_backfill_debt(repo_root)
    assert isinstance(result, list)
    return result


def test_absent_canonical_equations_package_is_clean(tmp_path: Path) -> None:
    assert _violations(tmp_path) == []


def test_parsable_noncandidate_modules_are_ignored(tmp_path: Path) -> None:
    _write_module(tmp_path, "VALUE = 1\n", "ordinary.py")
    _write_module(
        tmp_path,
        dedent(
            """
            def emits_label(path):
                return dict(canonical_producers=(path,))
            """
        ),
        "producer_only.py",
    )
    _write_module(
        tmp_path,
        dedent(
            """
            SOURCE_INPUT_SHA256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"

            def provenance_only(input_path):
                provenance = build_provenance_for_research_sidecar(sidecar_path=input_path)
                return provenance.source_sha256 == SOURCE_INPUT_SHA256
            """
        ),
        "provenance_only.py",
    )

    assert _violations(tmp_path) == []


def test_syntax_error_is_fail_closed_even_when_candidate_status_is_unknown(tmp_path: Path) -> None:
    path = _write_module(tmp_path, "def broken(:\n")

    violations = _violations(tmp_path)

    assert len(violations) == 1
    assert path.name in violations[0]
    assert "canonical producer source is unreadable" in violations[0]


def test_clean_exact_helper_and_sha_recheck_pass(tmp_path: Path) -> None:
    _write_module(tmp_path, _module_source())

    assert _violations(tmp_path) == []


def test_every_path_argument_must_be_routed_through_exact_helper(tmp_path: Path) -> None:
    _write_module(tmp_path, _module_source(builder=_MULTI_BUILDER))

    assert _violations(tmp_path) == []


def test_one_unrouted_path_argument_is_reported_by_name(tmp_path: Path) -> None:
    builder = dedent(
        """
        def build_contract(
            measurement_path=SOURCE_MEASUREMENT,
            frontier_path=SOURCE_FRONTIER,
        ):
            measurement_input, measurement_label = _canonical_producer_reference(
                measurement_path,
                canonical_path=SOURCE_MEASUREMENT,
            )
            frontier_input, frontier_label = frontier_path, SOURCE_FRONTIER
            measurement_provenance = build_provenance_for_research_sidecar(
                sidecar_path=measurement_input
            )
            frontier_provenance = build_provenance_for_research_sidecar(
                sidecar_path=frontier_input
            )
            if measurement_provenance.source_sha256 != SOURCE_MEASUREMENT_SHA256:
                raise ValueError("measurement hash drift")
            if frontier_provenance.source_sha256 != SOURCE_FRONTIER_SHA256:
                raise ValueError("frontier hash drift")
            return dict(canonical_producers=(measurement_label, frontier_label))
        """
    )
    _write_module(tmp_path, _module_source(builder=builder))

    violations = _violations(tmp_path)

    assert len(violations) == 1
    assert "frontier_path bypasses exact canonical-path binding" in violations[0]
    assert "measurement_path bypasses" not in violations[0]


def test_missing_exact_helper_is_rejected(tmp_path: Path) -> None:
    _write_module(tmp_path, _module_source(helper=""))

    assert "missing fail-closed exact-path helper" in _violations(tmp_path)[0]


def test_resolve_only_helper_cannot_launder_aliases(tmp_path: Path) -> None:
    resolve_only_helper = dedent(
        """
        def _canonical_producer_reference(value, *, canonical_path):
            resolved = (REPO_ROOT / value).resolve()
            expected = (REPO_ROOT / canonical_path).resolve()
            if resolved != expected:
                raise ValueError("canonical producer alias refused")
            return expected, canonical_path
        """
    )
    _write_module(tmp_path, _module_source(helper=resolve_only_helper))

    assert "missing fail-closed exact-path helper" in _violations(tmp_path)[0]


def test_helper_token_copy_that_ignores_value_is_rejected(tmp_path: Path) -> None:
    ignores_value_helper = _EXACT_HELPER.replace(
        "def _canonical_producer_reference(value, *, canonical_path):\n    candidate = Path(value)",
        "def _canonical_producer_reference(value, *, canonical_path):\n    candidate = Path(canonical_path)",
    )
    _write_module(tmp_path, _module_source(helper=ignores_value_helper))

    assert _violations(tmp_path)


def test_helper_call_without_canonical_path_keyword_is_rejected(tmp_path: Path) -> None:
    builder = dedent(
        """
        def build_contract(measurement_path=SOURCE_MEASUREMENT):
            input_path, source_label = _canonical_producer_reference(measurement_path)
            provenance = build_provenance_for_research_sidecar(sidecar_path=input_path)
            if provenance.source_sha256 != SOURCE_MEASUREMENT_SHA256:
                raise ValueError("hash drift")
            return dict(canonical_producers=(source_label,))
        """
    )
    _write_module(tmp_path, _module_source(builder=builder))

    assert "measurement_path bypasses exact canonical-path binding" in _violations(tmp_path)[0]


def test_guarded_path_must_feed_provenance_sink(tmp_path: Path) -> None:
    builder = _SINGLE_BUILDER.replace("sidecar_path=input_path", "sidecar_path=measurement_path")
    _write_module(tmp_path, _module_source(builder=builder))

    assert "measurement_path caller value leaks outside its exact-path guard" in _violations(tmp_path)[0]


def test_guarded_label_must_feed_canonical_producers(tmp_path: Path) -> None:
    builder = _SINGLE_BUILDER.replace(
        "canonical_producers=(source_label,)",
        "canonical_producers=(measurement_path,)",
    )
    _write_module(tmp_path, _module_source(builder=builder))

    assert "canonical_producers does not exactly match guarded labels" in _violations(tmp_path)[0]


def test_provenance_sha_must_be_rechecked_after_build(tmp_path: Path) -> None:
    builder = dedent(
        """
        def build_contract(measurement_path=SOURCE_MEASUREMENT):
            input_path, source_label = _canonical_producer_reference(
                measurement_path,
                canonical_path=SOURCE_MEASUREMENT,
            )
            provenance = build_provenance_for_research_sidecar(sidecar_path=input_path)
            frozen_sha = SOURCE_MEASUREMENT_SHA256
            return dict(canonical_producers=(source_label,))
        """
    )
    _write_module(tmp_path, _module_source(builder=builder))

    assert "producer SHA guards do not exactly cover" in _violations(tmp_path)[0]


def test_sha_equality_with_pass_is_not_fail_closed(tmp_path: Path) -> None:
    builder = _SINGLE_BUILDER.replace(
        """if provenance.source_sha256 != SOURCE_MEASUREMENT_SHA256:
        raise ValueError("hash drift")""",
        """if provenance.source_sha256 == SOURCE_MEASUREMENT_SHA256:
        pass""",
    )
    _write_module(tmp_path, _module_source(builder=builder))

    assert "producer SHA guard is not an exact immediate fail-closed comparison" in _violations(tmp_path)[0]


def test_wrong_provenance_to_sha_pairing_is_rejected(tmp_path: Path) -> None:
    builder = _MULTI_BUILDER.replace(
        "measurement_provenance.source_sha256 != SOURCE_MEASUREMENT_SHA256",
        "measurement_provenance.source_sha256 != SOURCE_FRONTIER_SHA256",
    ).replace(
        "frontier_provenance.source_sha256 != SOURCE_FRONTIER_SHA256",
        "frontier_provenance.source_sha256 != SOURCE_MEASUREMENT_SHA256",
    )
    _write_module(tmp_path, _module_source(builder=builder))

    assert "producer SHA guards do not exactly cover" in _violations(tmp_path)[0]


def test_parameter_renamed_file_cannot_evade_scope(tmp_path: Path) -> None:
    builder = dedent(
        """
        def build_contract(measurement_file=SOURCE_MEASUREMENT):
            provenance = build_provenance_for_research_sidecar(
                sidecar_path=measurement_file
            )
            if provenance.source_sha256 != SOURCE_MEASUREMENT_SHA256:
                raise ValueError("hash drift")
            return dict(canonical_producers=(measurement_file,))
        """
    )
    _write_module(tmp_path, _module_source(builder=builder))

    assert "measurement_file bypasses exact canonical-path binding" in _violations(tmp_path)[0]


def test_nested_canonical_equation_module_cannot_evade_scope(tmp_path: Path) -> None:
    _write_module(tmp_path, _module_source(helper=""), "nested/candidate.py")

    assert _violations(tmp_path)


def test_local_provenance_builder_shadow_is_rejected(tmp_path: Path) -> None:
    local_shadow = dedent(
        """
        def build_provenance_for_research_sidecar(*, sidecar_path):
            return object()
        """
    )
    _write_module(tmp_path, _module_source(builder=local_shadow + _SINGLE_BUILDER))

    assert "provenance builder is not the canonical unshadowed import" in _violations(tmp_path)[0]


def test_local_provenance_builder_assignment_shadow_is_rejected(tmp_path: Path) -> None:
    assignment_shadow = "build_provenance_for_research_sidecar = lambda **kwargs: object()\n"
    _write_module(tmp_path, _module_source(builder=assignment_shadow + _SINGLE_BUILDER))

    assert "provenance builder is not the canonical unshadowed import" in _violations(tmp_path)[0]


def test_substantive_same_line_waiver_suppresses_candidate(tmp_path: Path) -> None:
    broken_builder = _SINGLE_BUILDER.replace(
        "def build_contract(measurement_path=SOURCE_MEASUREMENT):",
        "def build_contract(measurement_path=SOURCE_MEASUREMENT):  # CANONICAL_PRODUCER_IDENTITY_OK:legacy external identity is audited upstream",
    )
    _write_module(tmp_path, _module_source(helper="", builder=broken_builder))

    assert _violations(tmp_path) == []


def test_placeholder_same_line_waiver_does_not_suppress_defect(tmp_path: Path) -> None:
    broken_builder = _SINGLE_BUILDER.replace(
        "def build_contract(measurement_path=SOURCE_MEASUREMENT):",
        "def build_contract(measurement_path=SOURCE_MEASUREMENT):  # CANONICAL_PRODUCER_IDENTITY_OK:TODO",
    )
    _write_module(tmp_path, _module_source(helper="", builder=broken_builder))

    assert "missing fail-closed exact-path helper" in _violations(tmp_path)[0]


def test_public_wrapper_warn_only_returns_defects(tmp_path: Path) -> None:
    _write_module(tmp_path, _module_source(helper=""))

    violations = preflight.check_evidence_authority_claims_producer_identity_backfill_ready(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "Rule chain: Catalog #351" in violations[0]


def test_public_wrapper_strict_mode_raises_for_future_flip(tmp_path: Path) -> None:
    _write_module(tmp_path, _module_source(helper=""))

    with pytest.raises(preflight.PreflightError, match="Catalog #351 backfill"):
        preflight.check_evidence_authority_claims_producer_identity_backfill_ready(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )


def test_backfill_scanner_reports_denominator(tmp_path: Path) -> None:
    _write_module(tmp_path, _module_source(), "a.py")
    _write_module(tmp_path, "VALUE = 1\n", "nested/b.py")

    violations, denominator = preflight._check_351_canonical_producer_identity_backfill_debt(
        tmp_path,
        include_denominator=True,
    )

    assert violations == []
    assert denominator == 2


def test_preflight_all_wires_backfill_guard_warn_only() -> None:
    source = inspect.getsource(preflight.preflight_all)

    assert "check_evidence_authority_claims_producer_identity_backfill_ready" in source
    assert "strict=False, verbose=verbose" in source
