from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tac.optimization.mdl_member_carrier_preflight import (
    MdlMemberCarrierPreflightConfigV1,
    MdlMemberCarrierPreflightProgramV1,
    audit_mdl_member_carrier,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(payload)
    return _sha(payload)


def _fixture(tmp_path: Path, *, stage_count: int = 2) -> MdlMemberCarrierPreflightConfigV1:
    stage_root = tmp_path / "stages"
    stage_root.mkdir()
    config_sha = "1" * 64
    for pair in range(stage_count):
        _write_json(
            stage_root / f"pair_{pair:04d}.json",
            {
                "schema": "mdl_polytope_member_pair_stage.v2",
                "pair_index": pair,
                "config_sha256": config_sha,
                "selected_equals_canonical": True,
                "changed_values": 0,
                "selected_frame_payload": None,
            },
        )
    full = tmp_path / "receipt_n64.json"
    full_sha = _write_json(
        full,
        {
            "schema": "mdl_polytope_member_measurement.v1",
            "completed_prefix": stage_count,
            "config_sha256": config_sha,
            "runtime": {"stage_root": str(stage_root)},
            "D2_exact_member_selection": {
                "selected_equals_canonical_pairs": stage_count,
                "integer_resize_exact_pairs": stage_count,
            },
            "D3_same_coder_comparison": {
                "selected_member_zlib9_bytes": 2000,
                "canonical_member_zlib9_bytes": 2000,
                "seed_coder": {
                    "scope": "seed coder is counted; raw-member zlib is diagnostic."
                },
            },
            "D4_n600_estimate_and_rate_feed": {"activated": False, "n600_estimate": None},
        },
    )
    compact = tmp_path / "compact.json"
    compact_sha = _write_json(compact, {"schema": "mdl_polytope_member_solve_compact_receipt.v1"})
    target = tmp_path / "target.json"
    target_sha = _write_json(
        target,
        {"schema": "direct_description_full_precision_target_planes.v1", "plane_dtype": "uint8"},
    )
    solver = tmp_path / "solver.py"
    solver.write_text("member_dtype = 'uint8'\n")
    producer = tmp_path / "producer.py"
    producer.write_text("selected_frame_payload = None\n")
    return MdlMemberCarrierPreflightConfigV1(
        compact_receipt_path=str(compact),
        compact_receipt_sha256=compact_sha,
        full_receipt_path=str(full),
        full_receipt_sha256=full_sha,
        stage_root=str(stage_root),
        target_receipt_path=str(target),
        target_receipt_sha256=target_sha,
        solver_source_path=str(solver),
        solver_source_sha256=_sha(solver.read_bytes()),
        producer_tool_path=str(producer),
        producer_tool_sha256=_sha(producer.read_bytes()),
    )


def test_preflight_scopes_missing_member_payload_without_curve(tmp_path: Path) -> None:
    result = audit_mdl_member_carrier(_fixture(tmp_path))

    assert result["verdict"] == "BLOCKED_602_OUTPUT_IS_NOT_A_RECEIVER_CARRIER"
    assert result["verdict_scope"].startswith("FORMULATION_OUTPUT_INTERFACE")
    assert result["curve"] == []
    assert result["source_output"]["selected_frame_payload_rows"] == 0
    assert result["eligibility_gates"]["pre_uint8_member_state"]["passed"] is False
    assert result["non_curve_diagnostic"]["registerable_curve_row"] is False
    assert result["pointer_moved"] is False


def test_preflight_rejects_stage_receipt_coverage_drift(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    stage = Path(config.stage_root) / "pair_0001.json"
    row = json.loads(stage.read_text())
    row["pair_index"] = 7
    _write_json(stage, row)

    with pytest.raises(DirectDescriptionError, match="stage identity drift"):
        audit_mdl_member_carrier(config)


def test_typed_program_compiles_only_read_only_consumer_argv() -> None:
    argv = MdlMemberCarrierPreflightProgramV1(
        config_path="config.json",
        output_path="receipt.json",
    ).compile_consumer_argv()

    assert argv.count("--config") == 1
    assert argv[-2:] == ("--execution-allowed", "false")
