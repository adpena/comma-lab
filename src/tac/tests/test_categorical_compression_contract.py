from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.categorical_compression_contract import build_categorical_compression_contract

REPO = Path(__file__).resolve().parents[3]


def test_categorical_contract_pins_class_order_and_grayscale_codebook() -> None:
    contract = build_categorical_compression_contract()

    assert contract["score_claim"] is False
    assert contract["dispatch_attempted"] is False
    assert contract["ready_for_exact_eval_dispatch"] is False
    assert [row["name"] for row in contract["classes"]] == [
        "road",
        "lane_markings",
        "undrivable",
        "movable",
        "my_car",
    ]
    assert [row["selfcomp_gray"] for row in contract["classes"]] == [0, 255, 64, 192, 128]
    assert [row["default_quant_bits"] for row in contract["classes"]] == [8, 8, 4, 6, 4]


def test_categorical_contract_requires_charged_runtime_consumption() -> None:
    contract = build_categorical_compression_contract()

    charged = contract["charged_byte_contract"]
    assert charged["every_decoder_table_is_archive_member"] is True
    assert charged["every_label_remap_is_archive_member"] is True
    assert charged["sidecars_outside_archive_forbidden"] is True
    families = contract["conditioning_families"]
    assert families["clade_spade"]["parameters_must_be_charged"] is True
    assert families["openpilot_priors"]["allowed_uncharged_use"] == (
        "compression_time_atom_ranking_only"
    )
    assert "runtime_consumes_conditioning_control" in contract["no_op_controls"]
    assert "requires_byte_closed_decoder_or_runtime_consumer" in contract["dispatch_blockers"]


def test_audit_categorical_compression_contract_cli_records_tool_manifest(tmp_path: Path) -> None:
    out = tmp_path / "categorical_contract.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_categorical_compression_contract.py"),
            "--json-out",
            str(out),
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["kind"] == "categorical_compression_contract"
    assert payload["tool_run_manifest"]["tool"] == "tools/audit_categorical_compression_contract.py"
    assert payload["tool_run_manifest"]["input_files"] == []
    assert payload["tool_run_manifest"]["score_claim"] is False
