from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO / "tools/probe_weight_decay_plasticity_ab.py"


def _load_tool():
    name = "_test_probe_weight_decay_plasticity_ab"
    spec = importlib.util.spec_from_file_location(name, TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_tool()


def _preregistration(control_wd: float = 0.0, treatment_wd: float = 0.01) -> dict:
    payload = {
        "schema_version": probe.PREREGISTRATION_SCHEMA,
        "immutable": True,
        "only_variable": "weight_decay",
        "control": {
            "arm_id": "control",
            "weight_decay": control_wd,
            "weight_decay_provenance": "operator-pinned control value",
        },
        "treatment": {
            "arm_id": "treatment",
            "weight_decay": treatment_wd,
            "weight_decay_provenance": "operator-pinned treatment value",
        },
    }
    payload["content_address_sha256"] = probe.canonical_sha256(payload)
    return payload


def _receipt(arm_id: str, preregistration_sha256: str, weight_decay: float, *, rank_kind: str = "effective_rank") -> dict:
    custody = {
        "authority": {"cohort": "real-n600", "pair_count": 600},
        "seed": "19",
        "pair_order_sha256": "a" * 64,
        "model_definition_sha256": "b" * 64,
        "init_ema_sha256": "c" * 64,
        "optimizer_non_weight_decay_fingerprint": "adamw-beta-eps-groups-v1",
        "curriculum_fingerprint": "curriculum-v1",
        "data_fingerprint": "gt-n600-v1",
        "non_weight_decay_config_sha256": "d" * 64,
    }
    rows = []
    for update in range(4):
        by_class = {}
        for name in probe.CLASSES:
            decline = 0.10 if name in probe.RARE_CLASSES else 0.20
            by_class[name] = 1.0 - decline * update
        bytes_count = 6000 - 60 * update
        rows.append(
            {
                "update": update,
                "wall_time_seconds": float(update + 1),
                "d_seg_by_class": by_class,
                "overall_d_seg": 1.0 - 0.16 * update,
                "trunk_weight_rank": {
                    "kind": rank_kind,
                    "value": 10.0 - update,
                    "parameter_scope": "trunk_weights",
                },
                "archive_bytes": bytes_count,
                "archive_sha256": format(update, "x") * 64,
                "archive_rate_bytes_per_pair": bytes_count / 600.0,
            }
        )
    return {
        "schema_version": probe.INPUT_SCHEMA,
        "arm_id": arm_id,
        "preregistration_sha256": preregistration_sha256,
        "declared_changed_fields": ["weight_decay"],
        "treatment": {"weight_decay": weight_decay},
        "custody": custody,
        "trajectory": rows,
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _arm_paths(tmp_path: Path) -> tuple[Path, Path, Path, dict, dict]:
    preregistration = _preregistration()
    preregistration_path = tmp_path / "preregistration.json"
    _write(preregistration_path, preregistration)
    preregistration_sha256 = probe.sha256_bytes(preregistration_path.read_bytes())
    control = _receipt("control", preregistration_sha256, 0.0)
    treatment = _receipt("treatment", preregistration_sha256, 0.01)
    control_path, treatment_path = tmp_path / "control.json", tmp_path / "treatment.json"
    _write(control_path, control)
    _write(treatment_path, treatment)
    return preregistration_path, control_path, treatment_path, control, treatment


def test_admits_matched_real_n600_receipts_and_reports_rank_rate_tradeoff(tmp_path: Path) -> None:
    preregistration_path, control_path, treatment_path, _, treatment = _arm_paths(tmp_path)
    for row in treatment["trajectory"]:
        for name in probe.RARE_CLASSES:
            row["d_seg_by_class"][name] -= 0.05 * row["update"]
        row["overall_d_seg"] -= 0.02 * row["update"]
        row["trunk_weight_rank"]["value"] -= 0.5 * row["update"]
        row["archive_bytes"] -= 120 * row["update"]
        row["archive_rate_bytes_per_pair"] = row["archive_bytes"] / 600.0
    _write(treatment_path, treatment)

    result, admitted = probe.analyze_receipts(preregistration_path, control_path, treatment_path)

    assert admitted
    assert result["owed_status"] == "CLOSED"
    assert result["launch_authorized"] is False
    assert result["verdict_scope"] == "INSTANCE"
    assert result["comparison"]["rare_class_keeps_learning"]["Lane"]["treatment"]["keeps_learning"]
    assert result["comparison"]["rank_rate_tradeoff"]["end_trunk_weight_rank_delta_treatment_minus_control"] < 0
    assert result["rates"]["treatment"]["archive"]["end_sha256"] == format(3, "x") * 64
    assert result["source_sha256"]["control"] == probe.sha256_bytes(control_path.read_bytes())
    content_address = result.pop("content_address_sha256")
    assert content_address == probe.canonical_sha256(result)


@pytest.mark.parametrize(
    ("mutator", "needle"),
    [
        (lambda prereg, control, treatment: control["custody"].update({"data_fingerprint": "other-data"}), "data_fingerprint"),
        (lambda prereg, control, treatment: treatment.update({"declared_changed_fields": ["weight_decay", "lr"]}), "declared_changed_fields"),
        (lambda prereg, control, treatment: treatment["treatment"].update({"weight_decay": 0.02}), "differs from preregistration"),
        (lambda prereg, control, treatment: treatment["trajectory"][0]["d_seg_by_class"].pop("Movable"), "canonical classes"),
        (lambda prereg, control, treatment: treatment["trajectory"][0]["trunk_weight_rank"].update({"parameter_scope": "codes"}), "never code rank"),
        (lambda prereg, control, treatment: treatment["trajectory"][0].pop("archive_sha256"), "archive_sha256"),
    ],
)
def test_rejects_any_non_weight_decay_difference_or_invalid_measurement(tmp_path: Path, mutator, needle: str) -> None:
    preregistration_path, control_path, treatment_path, control, treatment = _arm_paths(tmp_path)
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    mutator(preregistration, control, treatment)
    _write(control_path, control)
    _write(treatment_path, treatment)

    result, admitted = probe.analyze_receipts(preregistration_path, control_path, treatment_path)

    assert not admitted
    assert result["evidence_status"] == "BLOCKED_NO_EMPIRICAL_CLAIM"
    assert result["launch_authorized"] is False
    assert needle in result["blocker"]


def test_rejects_mutated_preregistration_content_address(tmp_path: Path) -> None:
    preregistration_path, control_path, treatment_path, _, _ = _arm_paths(tmp_path)
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    preregistration["treatment"]["weight_decay"] = 0.02
    _write(preregistration_path, preregistration)

    result, admitted = probe.analyze_receipts(preregistration_path, control_path, treatment_path)

    assert not admitted
    assert "does not address" in result["blocker"]


def test_rejects_rank_kind_mismatch(tmp_path: Path) -> None:
    preregistration_path, control_path, treatment_path, _, treatment = _arm_paths(tmp_path)
    for row in treatment["trajectory"]:
        row["d_seg_by_class"]["Lane"] = 1.0
        row["d_seg_by_class"]["Movable"] = 1.0
        row["trunk_weight_rank"]["kind"] = "pseudo_rank"
    _write(treatment_path, treatment)

    result, admitted = probe.analyze_receipts(preregistration_path, control_path, treatment_path)

    assert not admitted
    assert "rank kinds differ" in result["blocker"]


def test_rejects_rank_kind_changes_within_an_arm(tmp_path: Path) -> None:
    preregistration_path, control_path, treatment_path, _, treatment = _arm_paths(tmp_path)
    treatment["trajectory"][1]["trunk_weight_rank"]["kind"] = "pseudo_rank"
    _write(treatment_path, treatment)

    result, admitted = probe.analyze_receipts(preregistration_path, control_path, treatment_path)

    assert not admitted
    assert "must remain" in result["blocker"]


def test_rejects_one_archive_hash_with_multiple_byte_counts(tmp_path: Path) -> None:
    preregistration_path, control_path, treatment_path, _, treatment = _arm_paths(tmp_path)
    treatment["trajectory"][1]["archive_sha256"] = treatment["trajectory"][0]["archive_sha256"]
    _write(treatment_path, treatment)

    result, admitted = probe.analyze_receipts(preregistration_path, control_path, treatment_path)

    assert not admitted
    assert "multiple archive byte counts" in result["blocker"]


def test_reports_when_rare_classes_no_longer_keep_learning(tmp_path: Path) -> None:
    preregistration_path, control_path, treatment_path, _, treatment = _arm_paths(tmp_path)
    for row in treatment["trajectory"]:
        row["d_seg_by_class"]["Lane"] = 1.0
        row["d_seg_by_class"]["Movable"] = 1.0
    _write(treatment_path, treatment)

    result, admitted = probe.analyze_receipts(preregistration_path, control_path, treatment_path)

    assert admitted
    rare = result["comparison"]["rare_class_keeps_learning"]
    assert rare["Lane"]["treatment"]["keeps_learning"] is False
    assert rare["Movable"]["treatment"]["keeps_learning"] is False


def test_cli_writes_content_addressed_owed_receipt_for_missing_input(tmp_path: Path) -> None:
    output = tmp_path / "owed.json"
    rc = probe.main(
        [
            "--preregistration",
            str(tmp_path / "missing-preregistration.json"),
            "--control-receipt",
            str(tmp_path / "control.json"),
            "--treatment-receipt",
            str(tmp_path / "treatment.json"),
            "--output",
            str(output),
        ]
    )

    assert rc == 2
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["owed_status"] == "OWED"
    assert result["source_sha256"] == {"preregistration": None, "control": None, "treatment": None}
    address = result.pop("content_address_sha256")
    assert address == probe.canonical_sha256(result)
