from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "profile_pr85_qma9_context_entropy.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("profile_pr85_qma9_context_entropy_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_token_fixture(tmp_path: Path, tokens: np.ndarray, *, charged_bytes: int = 100) -> tuple[Path, Path]:
    token_path = tmp_path / "tokens.bin"
    token_path.write_bytes(tokens.astype(np.uint8, copy=False).tobytes(order="C"))
    sha = hashlib.sha256(token_path.read_bytes()).hexdigest()
    profile = {
        "decode": {"storage_order": "frame_major_header_width_by_header_height"},
        "mask_segment_identity": {"bytes": charged_bytes, "sha256": "mask-sha"},
        "schema": "test_profile",
        "token_source": {
            "dtype": "uint8",
            "range_contract": {"max": int(tokens.max()), "min": int(tokens.min())},
            "sha256": sha,
            "shape": [int(v) for v in tokens.shape],
        },
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return token_path, profile_path


def test_entropy_math_and_left_context_are_exact_for_synthetic_tokens(tmp_path: Path) -> None:
    module = _load_module()
    tokens = np.array(
        [
            [
                [0, 0],
                [1, 1],
                [0, 0],
                [1, 1],
            ]
        ],
        dtype=np.uint8,
    )
    token_path, profile_path = _write_token_fixture(tmp_path, tokens, charged_bytes=8)

    profile = module.build_profile(
        token_path=token_path,
        profile_json=profile_path,
        recorded_at_utc="2026-05-04T00:00:00Z",
    )

    assert module.entropy_from_counts(np.array([4, 4], dtype=np.int64)) == pytest.approx(1.0)
    assert profile["per_symbol_entropy"]["entropy_bits_per_token"] == pytest.approx(1.0)
    left = next(
        row
        for row in profile["conditional_context_entropy"]
        if row["name"] == "left_col_prev"
    )
    assert left["conditional_entropy_bits_per_token"] == pytest.approx(0.0)
    assert left["valid_token_count"] == 6
    assert left["excluded_token_count"] == 2
    assert profile["per_axis_entropy"]["cols"]["records"][0]["entropy_bits_per_token"] == pytest.approx(0.0)
    assert profile["per_axis_entropy"]["cols"]["records"][1]["entropy_bits_per_token"] == pytest.approx(0.0)


def test_profile_output_is_deterministic_and_never_claims_score_or_dispatch(tmp_path: Path) -> None:
    module = _load_module()
    tokens = np.array(
        [
            [[0, 1, 1], [0, 1, 1], [2, 2, 2]],
            [[0, 0, 1], [0, 1, 1], [2, 2, 1]],
        ],
        dtype=np.uint8,
    )
    token_path, profile_path = _write_token_fixture(tmp_path, tokens, charged_bytes=12)

    kwargs = {
        "token_path": token_path,
        "profile_json": profile_path,
        "recorded_at_utc": "2026-05-04T00:00:00Z",
    }
    first = module.build_profile(**kwargs)
    second = module.build_profile(**kwargs)

    first_json = json.dumps(first, sort_keys=True, allow_nan=False)
    second_json = json.dumps(second, sort_keys=True, allow_nan=False)
    assert first_json == second_json
    assert first["planning_only"] is True
    assert first["score_claim"] is False
    assert first["dispatch_performed"] is False
    assert first["gpu_required"] is False
    assert all(row["break_even_overhead_bytes"] == row["estimated_bytes_saved_lower_bound"] for row in first["opportunity_ranking"])
    markdown = module.render_markdown(first)
    assert "planning_only: true" in markdown
    assert "score_claim: false" in markdown
    assert "dispatch_performed: false" in markdown


def test_cli_writes_planning_artifacts(tmp_path: Path) -> None:
    module = _load_module()
    tokens = np.array([[[0, 0], [0, 1]], [[1, 1], [1, 0]]], dtype=np.uint8)
    token_path, profile_path = _write_token_fixture(tmp_path, tokens, charged_bytes=10)
    output_dir = tmp_path / "out"

    rc = module.main(
        [
            "--token-path",
            str(token_path),
            "--profile-json",
            str(profile_path),
            "--output-dir",
            str(output_dir),
            "--recorded-at-utc",
            "2026-05-04T00:00:00Z",
        ]
    )

    assert rc == 0
    payload = json.loads((output_dir / "pr85_qma9_context_entropy_profile.json").read_text())
    assert payload["schema"] == "pr85_qma9_context_entropy_profile_v1"
    assert payload["planning_only"] is True
    assert payload["score_claim"] is False
    assert payload["dispatch_performed"] is False
    assert (output_dir / "pr85_qma9_context_entropy_profile.md").exists()
