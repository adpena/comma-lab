# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "profile_pr85_residual_sufficient_program.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "profile_pr85_residual_sufficient_program_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_token_fixture(
    tmp_path: Path,
    storage_nwh: np.ndarray,
    *,
    charged_bytes: int = 64,
) -> tuple[Path, Path]:
    token_path = tmp_path / "tokens.bin"
    token_path.write_bytes(storage_nwh.astype(np.uint8, copy=False).tobytes(order="C"))
    profile = {
        "mask_segment_identity": {"bytes": charged_bytes, "sha256": "mask-sha"},
        "schema": "test_profile",
        "token_source": {
            "dtype": "uint8",
            "range_contract": {"max": int(storage_nwh.max()), "min": int(storage_nwh.min())},
            "sha256": hashlib.sha256(token_path.read_bytes()).hexdigest(),
            "shape": [int(v) for v in storage_nwh.shape],
        },
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return token_path, profile_path


def test_storage_tokens_are_normalized_to_render_order(tmp_path: Path) -> None:
    module = _load_module()
    storage_nwh = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3) % 5
    token_path, profile_path = _write_token_fixture(tmp_path, storage_nwh)

    storage, _profile = module.load_storage_tokens(token_path, profile_path)
    render = module.storage_nwh_to_render_nhw(storage)

    assert render.shape == (2, 3, 4)
    assert np.array_equal(render, np.transpose(storage_nwh, (0, 2, 1)))


def test_time_predictor_finds_exact_static_video_residual_program(tmp_path: Path) -> None:
    module = _load_module()
    render_nhw = np.array(
        [
            [[0, 1, 2], [3, 4, 0]],
            [[0, 1, 2], [3, 4, 0]],
            [[0, 1, 2], [3, 4, 0]],
        ],
        dtype=np.uint8,
    )
    storage_nwh = np.transpose(render_nhw, (0, 2, 1)).copy()
    token_path, profile_path = _write_token_fixture(tmp_path, storage_nwh, charged_bytes=32)

    profile = module.build_profile(
        token_path=token_path,
        profile_json=profile_path,
        predictors=("absolute_zero", "time_prev_zero_first"),
        recorded_at_utc="2026-05-04T00:00:00Z",
    )

    time_record = next(row for row in profile["residual_programs"] if row["predictor"] == "time_prev_zero_first")
    absolute_record = next(row for row in profile["residual_programs"] if row["predictor"] == "absolute_zero")
    assert time_record["zero_fraction"] > absolute_record["zero_fraction"]
    assert time_record["non_arbitrary_basis"].startswith("exact decoded PR85 QMA9")
    assert time_record["planning_only"] is True
    assert time_record["score_claim"] is False
    assert profile["dispatch_performed"] is False
    assert "render_order_sha256" in profile["input_token_source"]


def test_cli_writes_planning_artifacts(tmp_path: Path) -> None:
    module = _load_module()
    storage_nwh = np.array([[[0, 0], [1, 1]], [[0, 0], [1, 2]]], dtype=np.uint8)
    token_path, profile_path = _write_token_fixture(tmp_path, storage_nwh, charged_bytes=16)
    output_dir = tmp_path / "out"

    rc = module.main(
        [
            "--token-path",
            str(token_path),
            "--profile-json",
            str(profile_path),
            "--output-dir",
            str(output_dir),
            "--predictors",
            "absolute_zero,time_prev_zero_first",
            "--recorded-at-utc",
            "2026-05-04T00:00:00Z",
        ]
    )

    assert rc == 0
    payload = json.loads((output_dir / "pr85_residual_sufficient_program_profile.json").read_text())
    assert payload["schema"] == "pr85_residual_sufficient_program_profile_v1"
    assert payload["planning_only"] is True
    assert payload["score_claim"] is False
    assert (output_dir / "pr85_residual_sufficient_program_profile.md").exists()
