from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "remote_lane_line_search_c067.sh"


def _text() -> str:
    return SCRIPT.read_text()


def test_line_search_c067_script_is_cuda_and_custody_guarded() -> None:
    text = _text()
    assert "set -euo pipefail" in text
    assert "scripts/remote_archive_only_eval.sh" in text
    assert "scripts/probe_nvdec.sh" in text
    assert "--device cuda:0" in text
    assert "--device cuda" in text
    assert "contest_auth_eval.json" in text
    assert "ANCHOR_SHA=" in text
    assert "source_archive.zip SHA verified" in text
    executable_text = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "--output-json" not in executable_text
    assert "--device cpu" not in text
    assert "--device mps" not in text


def test_line_search_c067_script_accepts_reproducible_remote_paths() -> None:
    text = _text()
    for token in (
        "LS_RUN_ID",
        "LS_OUTPUT_DIR",
        "LS_SOURCE_ARCHIVE",
        "LS_SOURCE_METADATA",
        "LS_GT_VIDEO",
        "LS_POSENET_WEIGHTS",
    ):
        assert token in text
    assert 'LOG_DIR="${LS_OUTPUT_DIR:-$WORKSPACE/$LS_RUN_ID}"' in text


def test_line_search_c067_script_cleans_heavy_eval_work_after_json_copy() -> None:
    text = _text()
    assert 'cp "$EVAL_JSON" "$LOG_DIR/contest_auth_eval.json"' in text
    assert "LS_CLEAN_EVAL_WORK" in text
    assert '"$LOG_DIR/eval_work/inflated"' in text
    assert '"$LOG_DIR/eval_work/extracted"' in text
    assert '"$LOG_DIR/eval_work/archive.zip"' in text


def test_line_search_c067_script_exposes_nonarbitrary_search_knobs() -> None:
    text = _text()
    for token in (
        "LS_RADII",
        "LS_DELTA_SETS",
        "LS_GRADIENT_DELTA_SETS",
        "LS_GRADIENT_BACKTRACK_DELTAS",
        "LS_BASIS_DELTA_SETS",
        "LS_BASIS_MODES",
        "LS_BASIS_PAIR_INDICES",
        "LS_BASIS_WINDOW_RADIUS",
        "LS_BATCH_SIZE",
        "LS_CANDIDATE_CHUNK",
        "LS_MAX_CANDIDATE_ITEMS",
        "SEARCH_ARGS",
    ):
        assert token in text
    assert "--basis-delta-sets" in text
    assert "--gradient-delta-sets" in text
    assert "--delta-sets" in text
    assert "--max-candidate-items" in text
