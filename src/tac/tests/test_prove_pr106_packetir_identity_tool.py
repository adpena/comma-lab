# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "prove_pr106_packetir_identity.py"
PR106_R2_PR101_ARCHIVE = (
    REPO_ROOT / "submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip"
)
PR106_R2_PR101_SHA = "c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("prove_pr106_packetir_identity_tool", TOOL_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pr106_packetir_identity_tool_writes_nonpromotable_manifest(tmp_path: Path) -> None:
    tool = _load_tool()
    out = tmp_path / "identity.json"

    rc = tool.main(
        [
            "--archive",
            str(PR106_R2_PR101_ARCHIVE),
            "--expected-archive-sha256",
            PR106_R2_PR101_SHA,
            "--output-json",
            str(out),
        ]
    )

    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert manifest["schema"] == "pr106_sidecar_packet_ir_identity_proof_v1"
    assert manifest["packet_ir_identity_passed"] is True
    assert manifest["packet"]["format_id"] == "0x02"
    assert manifest["runtime_consumption_claim"] is False
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_pr106_packetir_identity_tool_fails_closed_on_expected_sha_mismatch(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    out = tmp_path / "identity.json"

    rc = tool.main(
        [
            "--archive",
            str(PR106_R2_PR101_ARCHIVE),
            "--expected-archive-sha256",
            "0" * 64,
            "--output-json",
            str(out),
        ]
    )

    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 2
    assert manifest["packet_ir_identity_passed"] is False
    assert manifest["blockers"] == ["expected_archive_sha256_mismatch"]
    assert manifest["score_claim"] is False
