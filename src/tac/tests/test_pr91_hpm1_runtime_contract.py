from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.pr91_hpm1_runtime_contract import (
    DEFAULT_PR91_RELEASE_RUNTIME_SOURCE_DIR,
    audit_pr91_hpm1_runtime_contract,
)

REPO = Path(__file__).resolve().parents[3]


def _write_runtime(source_dir: Path, *, inflate_device_expr: str, pr86_device_expr: str = '"cpu"') -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "inflate.py").write_text(
        "\n".join(
            [
                "from pr86_hpac import decompress_tokens_hpac as decompress_pr86_hpac_tokens",
                "def run(device):",
                "    return decompress_pr86_hpac_tokens(",
                "        b'tokens', 1, 2, 3, 'hpac.ppmd', 4, 5, 6,",
                f"        {inflate_device_expr}, False, 7,",
                "    )",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (source_dir / "pr86_hpac.py").write_text(
        "\n".join(
            [
                "def decompress_tokens_hpac(blob, N, H, W, hpac_pt, P, delta, ch, device, use_spm=False, hpac_d_film=32):",
                "    return None",
                "def standalone(device):",
                "    # Force HPAC decode onto CPU so it matches the CPU encoder bit-exactly.",
                "    return decompress_tokens_hpac(",
                "        b'tokens', 1, 2, 3, 'hpac.ppmd', 4, 5, 6,",
                f"        {pr86_device_expr}, False, 7,",
                "    )",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_pr91_runtime_contract_detects_cpu_comment_ambient_device_contradiction(
    tmp_path: Path,
) -> None:
    _write_runtime(tmp_path, inflate_device_expr="str(device)", pr86_device_expr="device")

    report = audit_pr91_hpm1_runtime_contract(source_dir=tmp_path)

    assert report["kind"] == "pr91_hpm1_runtime_contract"
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["ambient_device_call_count"] == 2
    assert report["hpac_device_contract"]["passed"] is False
    assert report["hpac_device_contract"]["resolved_device"] is None
    assert report["contradiction_count"] == 1
    assert report["contradictions"][0]["device_class"] == "ambient_device"
    assert "hpac_device_contract_resolved" in report["dispatch_blockers"]


def test_pr91_runtime_contract_accepts_literal_cpu_call_shape_but_still_blocks_dispatch(
    tmp_path: Path,
) -> None:
    _write_runtime(tmp_path, inflate_device_expr='"cpu"', pr86_device_expr='"cpu"')

    report = audit_pr91_hpm1_runtime_contract(source_dir=tmp_path)

    assert report["ambient_device_call_count"] == 0
    assert report["contradiction_count"] == 0
    assert {row["device_class"] for row in report["call_sites"]} == {"literal_cpu"}
    assert report["hpac_device_contract"]["passed"] is True
    assert report["hpac_device_contract"]["resolved_device"] == "cpu"
    assert report["ready_for_exact_eval_dispatch"] is False
    assert "hpac_device_contract_resolved" not in report["dispatch_blockers"]
    assert "runtime_consumer_sidecar_free_hpm1" in report["dispatch_blockers"]


@pytest.mark.skipif(
    not (DEFAULT_PR91_RELEASE_RUNTIME_SOURCE_DIR / "inflate.py").is_file()
    or not (DEFAULT_PR91_RELEASE_RUNTIME_SOURCE_DIR / "pr86_hpac.py").is_file(),
    reason="public PR91 release runtime sources not present",
)
def test_real_pr91_runtime_contract_records_ambient_hpac_device_use() -> None:
    report = audit_pr91_hpm1_runtime_contract()

    assert report["files"]["inflate_py"]["exists"] is True
    assert report["files"]["pr86_hpac_py"]["exists"] is True
    assert report["hpac_decoder_signature"]["device_arg_index"] == 8
    assert report["ambient_device_call_count"] >= 1
    assert report["ready_for_exact_eval_dispatch"] is False
    assert "hpac_device_contract_resolved" in report["dispatch_blockers"]


def test_audit_pr91_hpm1_runtime_contract_cli_records_tool_manifest(tmp_path: Path) -> None:
    _write_runtime(tmp_path / "runtime", inflate_device_expr="str(device)", pr86_device_expr="device")
    out = tmp_path / "runtime_contract.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_pr91_hpm1_runtime_contract.py"),
            "--source-dir",
            str(tmp_path / "runtime"),
            "--json-out",
            str(out),
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    tool_run = payload["tool_run_manifest"]
    assert payload["kind"] == "pr91_hpm1_runtime_contract"
    assert payload["dispatch_attempted"] is False
    assert tool_run["tool"] == "tools/audit_pr91_hpm1_runtime_contract.py"
    assert len(tool_run["input_files"]) == 2
