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
SCRIPT = REPO_ROOT / "experiments" / "export_pr85_residual_sufficient_program_curriculum.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "export_pr85_residual_sufficient_program_curriculum_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_profile_fixture(
    tmp_path: Path,
    render_nhw: np.ndarray,
    *,
    predictors: tuple[str, ...] = ("absolute_zero",),
) -> tuple[Path, Path]:
    storage_nwh = np.transpose(render_nhw.astype(np.uint8, copy=False), (0, 2, 1)).copy()
    token_path = tmp_path / "tokens_storage_nwh.bin"
    token_path.write_bytes(storage_nwh.tobytes(order="C"))
    profile = {
        "charged_baseline": {
            "mask_segment_bytes": 123,
            "mask_segment_sha256": "a" * 64,
        },
        "input_token_source": {
            "bytes": int(token_path.stat().st_size),
            "dtype": "uint8",
            "path": str(token_path),
            "render_order_sha256": _sha256(render_nhw.tobytes(order="C")),
            "sha256": _sha256(token_path.read_bytes()),
            "storage_shape": [int(value) for value in storage_nwh.shape],
        },
        "planning_only": True,
        "recorded_at_utc": "2026-05-04T00:00:00Z",
        "residual_programs": [
            {
                "best_lower_bound_bytes": float(index + 1),
                "nonzero_fraction": 0.1,
                "predictor": predictor,
                "zero_fraction": 0.9,
            }
            for index, predictor in enumerate(predictors)
        ],
        "schema": "fixture_pr85_residual_sufficient_program_profile_v1",
        "score_claim": False,
    }
    profile_path = tmp_path / "residual_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return profile_path, token_path


def test_build_export_computes_density_top_frames_and_capped_row_spans(tmp_path: Path) -> None:
    module = _load_module()
    render_nhw = np.array(
        [
            [[0, 0, 0, 0], [0, 1, 0, 0]],
            [[1, 1, 1, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [2, 0, 3, 4]],
        ],
        dtype=np.uint8,
    )
    profile_path, _token_path = _write_profile_fixture(tmp_path, render_nhw)

    summary, arrays = module.build_curriculum_export(
        profile_json=profile_path,
        predictor="absolute_zero",
        top_frame_count=2,
        max_row_spans=2,
    )

    assert summary["planning_only"] is True
    assert summary["score_claim"] is False
    assert summary["dispatch"] is False
    assert summary["dispatch_performed"] is False
    assert summary["predictor"]["id"] == "absolute_zero"
    assert summary["density"]["nonzero_count"] == 7
    assert summary["density"]["row_span_total_available"] == 3
    assert summary["density"]["row_span_record_count"] == 2
    assert [row["frame"] for row in summary["curriculum"]["top_frames"]] == [1, 2]
    assert [row["nonzero_count"] for row in summary["curriculum"]["top_frames"]] == [3, 3]
    assert summary["curriculum"]["row_span_records"] == [
        {"density": 0.75, "frame": 1, "nonzero_count": 3, "row": 0, "x0": 0, "x1": 2},
        {"density": 0.75, "frame": 2, "nonzero_count": 3, "row": 1, "x0": 0, "x1": 3},
    ]
    assert arrays["frame_nonzero_count"].tolist() == [1, 3, 3]
    assert arrays["row_nonzero_count"].tolist() == [[0, 1], [3, 0], [0, 3]]
    assert arrays["row_span_frame"].tolist() == [1, 2]
    assert arrays["row_span_row"].tolist() == [0, 1]
    assert arrays["row_span_x0"].tolist() == [0, 0]
    assert arrays["row_span_x1"].tolist() == [2, 3]


def test_cli_writes_deterministic_json_npz_with_custody(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_module()
    render_nhw = np.array(
        [
            [[0, 0, 0], [1, 0, 0]],
            [[2, 2, 0], [0, 3, 0]],
        ],
        dtype=np.uint8,
    )
    profile_path, _token_path = _write_profile_fixture(tmp_path, render_nhw)
    output_dir = tmp_path / "out"
    argv = [
        "--profile-json",
        str(profile_path),
        "--output-dir",
        str(output_dir),
        "--predictor",
        "absolute_zero",
        "--top-frame-count",
        "2",
        "--max-row-spans",
        "1",
    ]

    assert module.main(argv) == 0
    stdout = capsys.readouterr().out
    json_path = output_dir / "pr85_residual_sufficient_program_curriculum_density.json"
    npz_path = output_dir / "pr85_residual_sufficient_program_curriculum_density.npz"
    first_json = json_path.read_bytes()
    first_npz = npz_path.read_bytes()

    assert module.main(argv) == 0
    _stdout = capsys.readouterr().out
    assert json_path.read_bytes() == first_json
    assert npz_path.read_bytes() == first_npz
    payload = json.loads(first_json)
    assert payload["schema"] == "pr85_residual_sufficient_program_curriculum_density_v1"
    assert payload["planning_only"] is True
    assert payload["score_claim"] is False
    assert payload["dispatch"] is False
    assert payload["dispatch_performed"] is False
    assert payload["custody"]["npz"]["sha256"] == _sha256(first_npz)
    assert payload["artifact_contract"]["arrays_npz"]["row_density"]["shape"] == [2, 2]
    assert '"predictor_id": "absolute_zero"' in stdout

    with np.load(npz_path) as arrays:
        assert arrays["top_frame_index"].tolist() == [1, 0]
        assert arrays["frame_nonzero_count"].tolist() == [1, 3]
        assert arrays["row_span_frame"].tolist() == [1]
        assert arrays["row_span_row"].tolist() == [0]
        assert arrays["residual_symbol_counts"].tolist() == [8, 1, 2, 1, 0]


def test_unknown_profile_predictor_fails_closed(tmp_path: Path) -> None:
    module = _load_module()
    render_nhw = np.zeros((1, 2, 3), dtype=np.uint8)
    profile_path, _token_path = _write_profile_fixture(tmp_path, render_nhw)

    with pytest.raises(module.CurriculumExportError, match="not present in profile"):
        module.build_curriculum_export(
            profile_json=profile_path,
            predictor="left_zero_border",
        )
