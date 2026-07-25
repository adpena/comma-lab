from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.optimization.ddm_ga1_gauge_tolerance_ladder import (
    GA1GaugeToleranceError,
    compile_ga1_receipt,
    sha256_file,
)
from tools.build_ddm_ga1_gauge_tolerance_ladder import main as build_main

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".omx/research/configs/ddm_ga1_gauge_tolerance_ladder_20260725.json"


def test_current_sources_fail_closed_but_trigger_typed_mass_falsifier() -> None:
    receipt = compile_ga1_receipt(CONFIG, repository_root=ROOT)

    assert receipt["curve_admission"]["admitted"] is False
    assert receipt["curve_admission"]["curve_rows"] == []
    assert receipt["source_state"]["dr2b_grid"]["priced_sdwl1_rung_count"] == 0
    assert receipt["source_state"]["dr2b_grid"]["lossy_row_count"] == 0

    bound = receipt["current_c1_typed_mass_upper_bound"]
    assert bound["total_current_counted_bytes"] == 134_211
    assert bound["eligible_current_fiber_bytes"] == 151
    assert bound["maximum_convertible_fraction_of_current_counted_mass"] == pytest.approx(151 / 134_211)
    assert bound["falsifier_triggered_by_upper_bound"] is True
    assert bound["maximum_convertible_fraction_of_current_counted_mass"] < 0.05

    endpoint = receipt["source_state"]["endpoint_custody"]
    assert endpoint["box_errors"] == 136_839
    assert endpoint["delegated_exact_errors"] - endpoint["fresh_rd1_exact_errors"] == 4

    sense = receipt["costate_sense"]
    assert sense["row_count"] == 5
    assert all("class_index" not in row for row in sense["rows"])
    assert all(row["not_additive_across_class_strata"] for row in sense["rows"])
    assert receipt["rd1_crosscheck"]["actionable_dimension_duals"] == 0
    assert receipt["rd1_crosscheck"]["priced_effective_quanta"] == 0


def test_input_hash_drift_refuses_compilation(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_bytes())
    config["inputs"]["lp1"]["sha256"] = "0" * 64
    drifted = tmp_path / "config.json"
    drifted.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(GA1GaugeToleranceError, match="lp1: SHA-256 drift"):
        compile_ga1_receipt(drifted, repository_root=ROOT)


def test_equivalent_external_config_compiles_with_explicit_custody(tmp_path: Path) -> None:
    copied = tmp_path / "config.json"
    copied.write_bytes(CONFIG.read_bytes())

    receipt = compile_ga1_receipt(copied, repository_root=ROOT)

    assert receipt["typed_config"]["path"] == str(copied.resolve())
    assert receipt["typed_config"]["bytes"] == copied.stat().st_size
    assert receipt["typed_config"]["sha256"] == sha256_file(copied)


def test_materializer_refuses_output_outside_repository(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_bytes())
    escaped_output = tmp_path / "escaped.json"
    config["output_receipt"] = str(escaped_output)
    external = tmp_path / "external-config.json"
    external.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(SystemExit):
        build_main(["--config", str(external)])

    assert not escaped_output.exists()
