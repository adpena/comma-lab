from __future__ import annotations

from pathlib import Path

from tac.analysis.snerv_official_source_parity_audit import (
    _SNERV_FORWARD_PARITY_COMPONENT_SPECS,
    _component_state_rows,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_component_rows_consume_partial_forward_parity_artifact(tmp_path: Path) -> None:
    official_root = tmp_path / "official"
    local_root = tmp_path / "local"

    official_text_by_path: dict[str, list[str]] = {}
    local_text_by_path: dict[str, list[str]] = {}
    for spec in _SNERV_FORWARD_PARITY_COMPONENT_SPECS:
        for rel_path, marker in spec.get("official_source_markers") or ():
            official_text_by_path.setdefault(str(rel_path), []).append(str(marker))
        for rel_path, marker in spec.get("local_receiver_markers") or ():
            local_text_by_path.setdefault(str(rel_path), []).append(str(marker))
        for rel_path, marker in spec.get("primitive_parity_markers") or ():
            target = (
                official_text_by_path
                if str(rel_path).startswith("model/")
                else local_text_by_path
            )
            target.setdefault(str(rel_path), []).append(str(marker))

    for rel_path, markers in official_text_by_path.items():
        _write(official_root / rel_path, "\n".join(markers))
    for rel_path, markers in local_text_by_path.items():
        _write(local_root / rel_path, "\n".join(markers))

    official_group_rows = [
        {
            "group_id": spec["official_group_id"],
            "all_markers_present": True,
        }
        for spec in _SNERV_FORWARD_PARITY_COMPONENT_SPECS
    ]
    rows = _component_state_rows(
        official_root=official_root,
        local_root=local_root,
        official_group_rows=official_group_rows,
        local_receiver_safe_row={"all_markers_present": True},
        local_official_parity_row={"all_markers_present": False},
        forward_parity_artifact_row={
            "parity_passed": False,
            "parity_falsified": False,
            "component_rows": [
                {
                    "component_id": "mfu",
                    "classification": "official_source_fixture_mfu_state_dict_mapping_proven",
                    "source_forward_parity_proven": True,
                    "max_abs_error": 0.0,
                    "blockers": [],
                },
                {
                    "component_id": "hfr",
                    "classification": "official_source_fixture_hfr_state_dict_mapping_proven",
                    "source_forward_parity_proven": True,
                    "max_abs_error": 0.0,
                    "blockers": [],
                },
                {
                    "component_id": "tub",
                    "classification": "official_tub_graph_input_and_output2_fusion_source_fixture_proven_full_tub_blocked",
                    "source_forward_parity_proven": False,
                    "source_forward_parity_falsified": False,
                    "max_abs_error": 0.0,
                    "blockers": [
                        "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing"
                    ],
                },
            ],
        },
    )

    by_component = {row["component_id"]: row for row in rows}

    assert by_component["mfu"]["source_forward_parity_proven"] is True
    assert by_component["mfu"]["classification"] == (
        "official_source_fixture_mfu_state_dict_mapping_proven"
    )
    assert by_component["mfu"]["forward_parity_artifact_component_max_abs_error"] == 0.0
    assert "snerv_official_forward_parity_artifact_missing_or_failed" not in (
        by_component["mfu"]["blockers"]
    )
    assert "snerv_mfu_local_source_forward_markers_missing" not in (
        by_component["mfu"]["blockers"]
    )

    assert by_component["hfr"]["source_forward_parity_proven"] is True
    assert by_component["hfr"]["classification"] == (
        "official_source_fixture_hfr_state_dict_mapping_proven"
    )
    assert "snerv_hfr_local_source_forward_markers_missing" not in (
        by_component["hfr"]["blockers"]
    )

    assert by_component["tub"]["source_forward_parity_proven"] is False
    assert by_component["tub"]["source_forward_parity_falsified"] is False
    assert by_component["tub"]["classification"] == (
        "official_tub_graph_input_and_output2_fusion_source_fixture_proven_full_tub_blocked"
    )
    assert "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing" in (
        by_component["tub"]["blockers"]
    )
