# SPDX-License-Identifier: MIT
"""Adversarial tests for the Catalog #344 anchor round-trip extension."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tac.canonical_equations.equation import (
    CANONICAL_EQUATION_SCHEMA_VERSION,
    RECALIBRATE_ON_NEW_ANCHORS,
    VALID_EMPIRICAL_VERIFICATION_STATUSES,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.registry import (
    audit_empirical_anchor_roundtrip_fidelity,
)
from tac.preflight import (
    PreflightError,
    _check_344_anchor_roundtrip_integrity,
    check_empirical_finding_memo_references_canonical_equation,
)
from tac.provenance.builders import build_provenance_for_predicted


def _provenance():
    return build_provenance_for_predicted(
        model_id="catalog_344_roundtrip_fixture_v1",
        inputs_sha256="3" * 64,
        captured_at_utc="2026-07-20T00:00:00Z",
    )


def _anchor(
    anchor_id: str = "roundtrip_anchor_v1",
    *,
    status: str | None = None,
    noise_floor: float | None = None,
) -> EmpiricalAnchor:
    return EmpiricalAnchor(
        anchor_id=anchor_id,
        measurement_utc="2026-07-20T00:01:00Z",
        inputs={"x": 2.0, "nested": {"axis": "synthetic"}},
        predicted_output={"y": [1.0, 2.0]},
        empirical_output={"y": [1.1, 2.1]},
        residual=0.1,
        source_artifact="experiments/results/catalog_344_roundtrip_fixture",
        measurement_method="synthetic_roundtrip",
        provenance=_provenance(),
        empirical_verification_status=status,
        noise_floor=noise_floor,
        noise_floor_provenance=("deterministic synthetic fixture bound" if noise_floor is not None else None),
    )


def _equation(
    *anchors: EmpiricalAnchor,
    equation_id: str = "catalog_344_roundtrip_fixture_v1",
) -> CanonicalEquation:
    return CanonicalEquation(
        equation_id=equation_id,
        name="Catalog 344 round-trip fixture",
        one_line_summary="Synthetic equation for lossless anchor reconstruction tests.",
        latex_form=r"y = x",
        python_callable_module_path="tac.tests.catalog_344_fixture:predict",
        domain_of_validity={"axis": "synthetic"},
        units_in={"x": "unitless"},
        units_out={"y": "unitless"},
        empirical_anchors=tuple(anchors),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-20T00:02:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.tests.catalog_344_roundtrip_consumer",),
        canonical_producers=(),
        provenance=_provenance(),
    )


def _event(
    equation: CanonicalEquation,
    *,
    event_type: str = "registered",
) -> dict[str, object]:
    return {
        "schema_version": CANONICAL_EQUATION_SCHEMA_VERSION,
        "event_type": event_type,
        "equation_id": equation.equation_id,
        "equation_payload": equation.to_dict(),
        "written_at_utc": "2026-07-20T00:03:00Z",
        "written_pid": 1,
        "written_host": "synthetic",
        "agent": "codex",
        "subagent_id": "tests_344",
        "notes": "synthetic Catalog #344 round-trip fixture",
    }


def _event_anchor(event: dict[str, object], index: int = 0) -> dict[str, object]:
    payload = event["equation_payload"]
    assert isinstance(payload, dict)
    anchors = payload["empirical_anchors"]
    assert isinstance(anchors, list)
    anchor = anchors[index]
    assert isinstance(anchor, dict)
    return anchor


def _write_registry(root: Path, events: list[dict[str, object]]) -> Path:
    path = root / ".omx" / "state" / "canonical_equations_registry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    return path


_STATIC_FIELDS = (
    "anchor_id: str",
    "empirical_verification_status: str | None = None",
)
_STATIC_MAPPINGS = (
    'anchor_id=payload["anchor_id"]',
    'empirical_verification_status=payload.get("empirical_verification_status")',
)


def _write_static_package(
    root: Path,
    *,
    fields: tuple[str, ...] = _STATIC_FIELDS,
    mappings: tuple[str, ...] = _STATIC_MAPPINGS,
    include_registry: bool = True,
) -> None:
    package = root / "src" / "tac" / "canonical_equations"
    package.mkdir(parents=True, exist_ok=True)
    model_source = "class EmpiricalAnchor:\n" + "".join(f"    {field}\n" for field in fields)
    (package / "equation.py").write_text(model_source, encoding="utf-8")
    if not include_registry:
        return
    registry_source = (
        "def reconstruct(payload):\n"
        "    return EmpiricalAnchor(\n" + "".join(f"        {mapping},\n" for mapping in mappings) + "    )\n"
    )
    (package / "registry.py").write_text(registry_source, encoding="utf-8")


@pytest.mark.parametrize(
    "status",
    sorted(VALID_EMPIRICAL_VERIFICATION_STATUSES),
)
def test_audit_roundtrips_each_empirical_verification_status(status: str) -> None:
    event = _event(_equation(_anchor(status=status)))

    assert _event_anchor(event)["empirical_verification_status"] == status
    assert audit_empirical_anchor_roundtrip_fidelity(rows=[event]) == ()


def test_audit_keeps_legacy_optional_fields_absent() -> None:
    event = _event(_equation(_anchor()))
    serialized = _event_anchor(event)

    assert "empirical_verification_status" not in serialized
    assert "noise_floor" not in serialized
    assert "noise_floor_provenance" not in serialized
    assert audit_empirical_anchor_roundtrip_fidelity(rows=[event]) == ()


def test_audit_roundtrips_noise_floor_and_its_provenance() -> None:
    event = _event(_equation(_anchor(noise_floor=0.025)))
    serialized = _event_anchor(event)

    assert serialized["noise_floor"] == pytest.approx(0.025)
    assert serialized["noise_floor_provenance"] == "deterministic synthetic fixture bound"
    assert audit_empirical_anchor_roundtrip_fidelity(rows=[event]) == ()


def test_audit_rejects_unknown_additive_anchor_key() -> None:
    event = _event(_equation(_anchor()))
    _event_anchor(event)["future_authority_class"] = {"tier": 7}

    violations = audit_empirical_anchor_roundtrip_fidelity(rows=[event])

    assert len(violations) == 1
    assert violations[0].anchor_index == 0
    assert violations[0].changed_json_paths == ("$[0].future_authority_class",)
    assert "future_authority_class" in violations[0].original
    assert "future_authority_class" not in violations[0].reconstructed


def test_audit_fails_closed_on_malformed_anchor() -> None:
    event = _event(_equation(_anchor()))
    _event_anchor(event).pop("measurement_method")

    violations = audit_empirical_anchor_roundtrip_fidelity(rows=[event])

    assert len(violations) == 1
    assert violations[0].anchor_index == -1
    assert violations[0].changed_json_paths == ("$",)
    assert "reconstruction_error" in violations[0].reconstructed


def test_audit_attributes_mismatch_to_only_tampered_anchor() -> None:
    event = _event(
        _equation(
            _anchor("first_anchor_v1"),
            _anchor("second_anchor_v1"),
        )
    )
    _event_anchor(event, 1)["future_scope"] = "second-only"

    violations = audit_empirical_anchor_roundtrip_fidelity(rows=[event])

    assert len(violations) == 1
    assert violations[0].anchor_index == 1
    assert violations[0].changed_json_paths == ("$[1].future_scope",)


def test_audit_visits_every_event_and_preserves_event_attribution() -> None:
    events = [
        _event(_equation(_anchor("registered_anchor_v1")), event_type="registered"),
        _event(
            _equation(
                _anchor("appended_anchor_v1"),
                equation_id="catalog_344_appended_fixture_v1",
            ),
            event_type="anchor_appended",
        ),
        _event(
            _equation(
                _anchor("recalibrated_anchor_v1"),
                equation_id="catalog_344_recalibrated_fixture_v1",
            ),
            event_type="recalibrated",
        ),
    ]
    _event_anchor(events[1])["future_axis"] = "appended"
    _event_anchor(events[2])["future_axis"] = "recalibrated"

    violations = audit_empirical_anchor_roundtrip_fidelity(rows=events)

    assert [(v.registry_line, v.event_type) for v in violations] == [
        (2, "anchor_appended"),
        (3, "recalibrated"),
    ]


def test_audit_reads_synthetic_jsonl_path(tmp_path: Path) -> None:
    good = _event(_equation(_anchor("path_good_anchor_v1")))
    tampered = copy.deepcopy(good)
    _event_anchor(tampered)["future_jsonl_key"] = True
    path = tmp_path / "registry.jsonl"
    path.write_text(
        json.dumps(good, sort_keys=True) + "\n" + json.dumps(tampered, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    violations = audit_empirical_anchor_roundtrip_fidelity(path)

    assert len(violations) == 1
    assert violations[0].registry_line == 2
    assert violations[0].changed_json_paths == ("$[0].future_jsonl_key",)


def test_static_valid_required_and_defaulted_mappings_are_clean(tmp_path: Path) -> None:
    _write_static_package(tmp_path)

    assert _check_344_anchor_roundtrip_integrity(tmp_path) == []


def test_static_missing_required_mapping_is_flagged(tmp_path: Path) -> None:
    _write_static_package(
        tmp_path,
        mappings=('empirical_verification_status=payload.get("empirical_verification_status")',),
    )

    violations = _check_344_anchor_roundtrip_integrity(tmp_path)

    assert len(violations) == 1
    assert "'anchor_id'" in violations[0]
    assert "dropped during registry reconstruction" in violations[0]


def test_static_wrong_required_field_mapping_is_flagged(tmp_path: Path) -> None:
    _write_static_package(
        tmp_path,
        mappings=(
            'anchor_id=payload["other_anchor_id"]',
            'empirical_verification_status=payload.get("empirical_verification_status")',
        ),
    )

    violations = _check_344_anchor_roundtrip_integrity(tmp_path)

    assert len(violations) == 1
    assert "'anchor_id'" in violations[0]
    assert "same-name payload read" in violations[0]


def test_static_defaulted_field_requires_legacy_compatible_get(tmp_path: Path) -> None:
    _write_static_package(
        tmp_path,
        mappings=(
            'anchor_id=payload["anchor_id"]',
            'empirical_verification_status=payload["empirical_verification_status"]',
        ),
    )

    violations = _check_344_anchor_roundtrip_integrity(tmp_path)

    assert len(violations) == 1
    assert "'empirical_verification_status'" in violations[0]
    assert "legacy-compatible payload.get" in violations[0]


def test_static_defaulted_field_accepts_same_name_get(tmp_path: Path) -> None:
    _write_static_package(
        tmp_path,
        fields=("empirical_verification_status: str | None = None",),
        mappings=('empirical_verification_status=payload.get("empirical_verification_status")',),
    )

    assert _check_344_anchor_roundtrip_integrity(tmp_path) == []


def test_static_substantive_same_line_waiver_allows_missing_mapping(
    tmp_path: Path,
) -> None:
    _write_static_package(
        tmp_path,
        fields=(
            "anchor_id: str",
            "empirical_verification_status: str | None = None  "
            "# CANONICAL_ANCHOR_ROUNDTRIP_OK:legacy reader intentionally remains external",
        ),
        mappings=('anchor_id=payload["anchor_id"]',),
    )

    assert _check_344_anchor_roundtrip_integrity(tmp_path) == []


def test_static_placeholder_same_line_waiver_is_rejected(tmp_path: Path) -> None:
    _write_static_package(
        tmp_path,
        fields=(
            "anchor_id: str",
            "empirical_verification_status: str | None = None  # CANONICAL_ANCHOR_ROUNDTRIP_OK:<rationale>",
        ),
        mappings=('anchor_id=payload["anchor_id"]',),
    )

    violations = _check_344_anchor_roundtrip_integrity(tmp_path)

    assert len(violations) == 1
    assert "'empirical_verification_status'" in violations[0]
    assert "dropped during registry reconstruction" in violations[0]


def test_static_missing_canonical_equations_package_skips_cleanly(
    tmp_path: Path,
) -> None:
    assert _check_344_anchor_roundtrip_integrity(tmp_path) == []


def test_static_missing_registry_reader_fails_closed(tmp_path: Path) -> None:
    _write_static_package(tmp_path, include_registry=False)

    violations = _check_344_anchor_roundtrip_integrity(tmp_path)

    assert len(violations) == 1
    assert "missing registry reader" in violations[0]


def test_public_strict_wrapper_raises_on_tampered_synthetic_ledger(
    tmp_path: Path,
) -> None:
    _write_static_package(tmp_path)
    event = _event(_equation(_anchor()))
    _event_anchor(event)["future_authority_class"] = "tampered"
    _write_registry(tmp_path, [event])

    with pytest.raises(PreflightError, match="Catalog #344"):
        check_empirical_finding_memo_references_canonical_equation(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )


def test_public_non_strict_wrapper_returns_dynamic_mismatch_without_research_dir(
    tmp_path: Path,
) -> None:
    _write_static_package(tmp_path)
    event = _event(_equation(_anchor()))
    _event_anchor(event)["future_authority_class"] = "tampered"
    _write_registry(tmp_path, [event])

    violations = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "future_authority_class" in violations[0]


def test_live_repo_anchor_roundtrip_scope_extension_has_zero_defects() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    assert _check_344_anchor_roundtrip_integrity(repo_root) == []
