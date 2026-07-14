from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO / "tools/probe_ordinal_perclass_convergence.py"


def _load_tool():
    name = "_test_probe_ordinal_perclass_convergence"
    spec = importlib.util.spec_from_file_location(name, TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_tool()


def _receipt(loss: str, *, margin: float | None = None, pairs: int = 600, seed: str = "19") -> dict:
    custody = {
        "authority": {"cohort": "real-n600", "pair_count": pairs},
        "seed": seed,
        "order_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "optimizer_fingerprint": "opt-v1",
        "curriculum_fingerprint": "curriculum-v1",
        "init_ema_sha256": "c" * 64,
        "non_treatment_config_sha256": "d" * 64,
        "data_fingerprint": "e" * 64,
        "preregistration_sha256": "f" * 64,
    }
    starts = {"Road": 1.0, "Lane": 1.0, "Undrivable": 1.0, "Movable": 1.0, "MyCar": 1.0}
    rows = []
    for update in range(4):
        classes = {}
        for item, start in starts.items():
            decline = 0.10 if item in {"Lane", "Movable"} else 0.20
            value = start - decline * update
            classes[item] = {"all": value, "hard": value + 0.1, "easy": value - 0.1}
        rows.append({"update": update, "wall_time_seconds": float(update + 1), "d_seg_by_class": classes})
    treatment = {"seg_loss": loss}
    if margin is not None:
        treatment["margin_target_end"] = margin
    return {"schema_version": probe.INPUT_SCHEMA, "custody": custody, "treatment": treatment, "trajectory": rows}


def _write(path: Path, receipt: dict) -> None:
    path.write_text(json.dumps(receipt), encoding="utf-8")


def test_real_n600_matched_margin_receipts_are_admitted_and_content_addressed(tmp_path: Path) -> None:
    ce = _receipt("ce")
    margin = _receipt("margin_hinge", margin=0.0)
    for row in margin["trajectory"]:
        for item in ("Lane", "Movable"):
            for stratum in ("all", "hard", "easy"):
                row["d_seg_by_class"][item][stratum] -= 0.06 * row["update"]
    ce_path, margin_path = tmp_path / "ce.json", tmp_path / "margin.json"
    _write(ce_path, ce)
    _write(margin_path, margin)

    result, admitted = probe.analyze_receipts(ce_path, margin_path)

    assert admitted
    assert result["owed_status"] == "CLOSED"
    assert result["launch_authorized"] is False
    assert result["comparison"]["instance_verdict"] == "DOMINANT"
    assert result["comparison"]["rare_common_gap"]["closure_percent"] == pytest.approx(60.0)
    assert result["source_receipt_sha256"]["ce"] == probe.sha256_bytes(ce_path.read_bytes())
    address = result.pop("content_address_sha256")
    assert address == probe.canonical_sha256(result)


@pytest.mark.parametrize(
    ("mutator", "needle"),
    [
        (lambda ce, margin: ce["custody"].update({"authority": {"cohort": "fixture", "pair_count": 600}}), "real-n600"),
        (lambda ce, margin: margin["custody"].update({"seed": "different"}), "custody mismatch"),
        (lambda ce, margin: margin["custody"].update({"data_fingerprint": "0" * 64}), "custody mismatch"),
        (lambda ce, margin: margin["treatment"].update({"margin_target_end": 0.1}), "margin_target_end"),
        (lambda ce, margin: margin["treatment"].update({"extra_knob": True}), "confounded treatment knobs"),
        (lambda ce, margin: ce["trajectory"][0]["d_seg_by_class"].pop("Movable"), "canonical classes"),
    ],
)
def test_invalid_custody_or_schema_is_owed_not_empirical(tmp_path: Path, mutator, needle: str) -> None:
    ce = _receipt("ce")
    margin = _receipt("margin_hinge", margin=0.0)
    mutator(ce, margin)
    ce_path, margin_path = tmp_path / "ce.json", tmp_path / "margin.json"
    _write(ce_path, ce)
    _write(margin_path, margin)

    result, admitted = probe.analyze_receipts(ce_path, margin_path)

    assert not admitted
    assert result["verdict_scope"] == "INSTANCE"
    assert result["evidence_status"] == "BLOCKED_NO_EMPIRICAL_CLAIM"
    assert result["owed_status"] == "OWED"
    assert result["launch_authorized"] is False
    assert needle in result["blocker"]


def test_tradeoff_when_common_or_stratum_rate_regresses(tmp_path: Path) -> None:
    ce = _receipt("ce")
    margin = _receipt("margin_hinge", margin=0.0)
    for row in margin["trajectory"]:
        # Faster rare improvement, but Road gets worse: exact tradeoff classification.
        row["d_seg_by_class"]["Lane"]["all"] += 0.08 * row["update"]
        row["d_seg_by_class"]["Road"]["hard"] += 0.01 * row["update"]
    ce_path, margin_path = tmp_path / "ce.json", tmp_path / "margin.json"
    _write(ce_path, ce)
    _write(margin_path, margin)

    result, admitted = probe.analyze_receipts(ce_path, margin_path)

    assert admitted
    assert result["comparison"]["instance_verdict"] == "TRADEOFF"


def test_cli_writes_machine_readable_owed_receipt_for_missing_input(tmp_path: Path) -> None:
    output = tmp_path / "owed.json"
    rc = probe.main(["--ce-receipt", str(tmp_path / "missing.json"), "--margin-receipt", str(tmp_path / "other.json"), "--output", str(output)])

    assert rc == 2
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["owed_status"] == "OWED"
    assert result["source_receipt_sha256"] == {"ce": None, "margin_hinge": None}
