# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.compile_ddm_menu1_postcharter_addendum import PostcharterConfig, run

CONFIG = Path(
    ".omx/research/configs/ddm_menu1_postcharter_addendum_20260724.json"
)


def test_checked_in_config_is_contained_and_compiles_deterministically(
    tmp_path: Path,
) -> None:
    config = PostcharterConfig.model_validate_json(CONFIG.read_bytes())
    assert config.execution_allowed is True
    assert config.research_only is True
    assert config.score_claim is False
    output = tmp_path / "receipt.json"
    run(CONFIG, output)
    first = output.read_bytes()
    run(CONFIG, output)
    assert output.read_bytes() == first
    receipt = json.loads(first)
    assert receipt["verdict"] == "MENU1_POSTCHARTER_JOINED_BOX_NOT_REACHED"
    assert receipt["box"]["r6_candidate"] is False
    assert receipt["main_landing_review_required"] is True


def test_input_hash_drift_refuses(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["mc1_receipt_sha256"] = "0" * 64
    bad_config = tmp_path / "bad.json"
    bad_config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="MC1 SHA-256 differs"):
        run(bad_config, tmp_path / "receipt.json")
