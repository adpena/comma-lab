# SPDX-License-Identifier: MIT
"""Adversarial coverage for the Catalog #351 producer-identity extension."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from textwrap import dedent

import pytest

import tac.canonical_equations.einstein_kolmogorov_crux_20260719 as crux_equation
from tac import preflight

_EXACT_HELPER = dedent(
    """
    def _contract_input_path(value):
        candidate = Path(value)
        return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()

    def _canonical_producer_reference(value, *, canonical_path):
        candidate = Path(value)
        requested = candidate.absolute() if candidate.is_absolute() else (REPO_ROOT / candidate).absolute()
        expected_requested = (REPO_ROOT / canonical_path).absolute()
        if requested != expected_requested:
            raise ValueError("canonical producer alias refused")
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
            from pathlib import Path

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
    package = repo_root / "src/tac/canonical_equations"
    package.mkdir(parents=True, exist_ok=True)
    path = package / name
    path.write_text(source, encoding="utf-8")
    return path


def _violations(repo_root: Path) -> list[str]:
    return preflight._check_351_canonical_producer_identity(repo_root)


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


def test_helper_that_returns_caller_alias_instead_of_expected_path_is_rejected(
    tmp_path: Path,
) -> None:
    alias_return_helper = _EXACT_HELPER.replace(
        "return expected, canonical_path",
        "return resolved, canonical_path",
    )
    _write_module(tmp_path, _module_source(helper=alias_return_helper))

    assert "missing fail-closed exact-path helper" in _violations(tmp_path)[0]


def test_resolve_only_helper_cannot_launder_symlink_aliases(tmp_path: Path) -> None:
    resolve_only_helper = dedent(
        """
        def _contract_input_path(value):
            candidate = Path(value)
            return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()

        def _canonical_producer_reference(value, *, canonical_path):
            resolved = _contract_input_path(value)
            expected = (REPO_ROOT / canonical_path).resolve()
            if resolved != expected:
                raise ValueError("canonical producer alias refused")
            return expected, canonical_path
        """
    )
    _write_module(tmp_path, _module_source(helper=resolve_only_helper))

    assert "missing fail-closed exact-path helper" in _violations(tmp_path)[0]


def test_helper_that_does_not_raise_on_path_mismatch_is_rejected(tmp_path: Path) -> None:
    non_fail_closed_helper = dedent(
        """
        def _contract_input_path(value):
            candidate = Path(value)
            return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()

        def _canonical_producer_reference(value, *, canonical_path):
            resolved = _contract_input_path(value)
            expected = (REPO_ROOT / canonical_path).resolve()
            if resolved != expected:
                resolved = expected
            return expected, canonical_path
        """
    )
    _write_module(tmp_path, _module_source(helper=non_fail_closed_helper))

    assert "missing fail-closed exact-path helper" in _violations(tmp_path)[0]


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
            return dict(canonical_producers=(source_label, frozen_sha))
        """
    )
    _write_module(tmp_path, _module_source(builder=builder))

    assert "SOURCE_MEASUREMENT_SHA256 is not checked against provenance.source_sha256" in _violations(tmp_path)[0]


def test_each_of_multiple_sha_constants_requires_its_own_recheck(tmp_path: Path) -> None:
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
            frontier_input, frontier_label = _canonical_producer_reference(
                frontier_path,
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
            expected_frontier_sha = SOURCE_FRONTIER_SHA256
            return dict(
                canonical_producers=(measurement_label, frontier_label, expected_frontier_sha)
            )
        """
    )
    _write_module(tmp_path, _module_source(builder=builder))

    violations = _violations(tmp_path)

    assert len(violations) == 1
    assert "SOURCE_FRONTIER_SHA256 is not checked" in violations[0]
    assert "SOURCE_MEASUREMENT_SHA256 is not checked" not in violations[0]


def test_multiple_sha_constants_all_revalidated_are_clean(tmp_path: Path) -> None:
    _write_module(tmp_path, _module_source(builder=_MULTI_BUILDER))

    assert _violations(tmp_path) == []


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

    violations = _violations(tmp_path)

    assert len(violations) == 1
    assert "missing fail-closed exact-path helper" in violations[0]


def test_public_wrapper_raises_preflight_error_for_producer_defect_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, _module_source(helper=""))
    monkeypatch.setattr(preflight, "_run_v9_provenance_gate", lambda *_args, **_kwargs: [])

    with pytest.raises(preflight.PreflightError, match="producer custody defects"):
        preflight.check_evidence_authority_claims_are_custodied(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )


def test_public_wrapper_returns_extension_defect_when_not_strict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, _module_source(helper=""))
    monkeypatch.setattr(preflight, "_run_v9_provenance_gate", lambda *_args, **_kwargs: [])

    violations = preflight.check_evidence_authority_claims_are_custodied(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "missing fail-closed exact-path helper" in violations[0]


def test_live_repository_has_zero_catalog_351_producer_identity_defects() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    assert _violations(repo_root) == []


def test_target_builder_defaults_are_repo_relative_not_cwd_relative(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    equation = crux_equation.build_einstein_kolmogorov_crux_action_rate_contract_v1()

    assert equation.provenance.source_path == crux_equation.SOURCE_MEASUREMENT
    assert equation.canonical_producers == (
        crux_equation.SOURCE_MEASUREMENT,
        crux_equation.SOURCE_FRONTIER_MAGNITUDE,
    )


@pytest.mark.parametrize(
    ("producer", "canonical_name"),
    [
        ("measurement", "SOURCE_MEASUREMENT"),
        ("frontier", "SOURCE_FRONTIER_MAGNITUDE"),
    ],
)
def test_target_builder_rejects_byte_identical_noncanonical_copy(
    producer: str,
    canonical_name: str,
    tmp_path: Path,
) -> None:
    canonical = crux_equation.REPO_ROOT / getattr(crux_equation, canonical_name)
    alias = tmp_path / f"{producer}-copy.json"
    shutil.copyfile(canonical, alias)
    kwargs = {
        "measurement_path": crux_equation.SOURCE_MEASUREMENT,
        "frontier_chart_path": crux_equation.SOURCE_FRONTIER_MAGNITUDE,
    }
    kwargs[f"{producer}_path" if producer == "measurement" else "frontier_chart_path"] = alias

    with pytest.raises(ValueError, match="canonical producer path must resolve to"):
        crux_equation.build_einstein_kolmogorov_crux_action_rate_contract_v1(**kwargs)


def test_target_builder_rejects_symlink_alias_where_supported(tmp_path: Path) -> None:
    canonical = crux_equation.REPO_ROOT / crux_equation.SOURCE_MEASUREMENT
    alias = tmp_path / "measurement-symlink.json"
    try:
        alias.symlink_to(canonical)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="canonical producer path must resolve to"):
        crux_equation.build_einstein_kolmogorov_crux_action_rate_contract_v1(measurement_path=alias)


def test_target_builder_rejects_hardlink_alias_where_supported(tmp_path: Path) -> None:
    canonical = crux_equation.REPO_ROOT / crux_equation.SOURCE_MEASUREMENT
    alias = tmp_path / "measurement-hardlink.json"
    try:
        os.link(canonical, alias)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="canonical producer path must resolve to"):
        crux_equation.build_einstein_kolmogorov_crux_action_rate_contract_v1(measurement_path=alias)
