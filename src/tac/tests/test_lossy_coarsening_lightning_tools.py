from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_script(relpath: str):
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lossy_coarsening_generated_inflate_uses_model_schema_once() -> None:
    module = _load_script("experiments/lossy_coarsening_lightning_cuda_test.py")

    inflate_source = module.FORKED_INFLATE_PY

    assert "def _fixed_state_schema()" in inflate_source
    assert "probe.state_dict().items()" in inflate_source
    assert '("blocks.0.weight", (36, 28, 1, 1))' not in inflate_source
    assert "reconstructed_q = chunk" in inflate_source
    assert "chunk * Ks" not in inflate_source


def test_lossy_coarsening_remote_command_uses_auth_eval_json_path() -> None:
    module = _load_script("experiments/lossy_coarsening_lightning_cuda_test.py")

    command = module.build_remote_command(
        job_name="job-test",
        remote_pact="/remote/pact",
        archive_relpath="experiments/results/lossy/archive.zip",
        submission_dir_relpath="experiments/results/lossy/submission_dir",
    )

    assert "experiments/contest_auth_eval.py" in command
    assert "--archive \"$ARCHIVE\"" in command
    assert "--inflate-sh \"$INFLATE_SH\"" in command
    assert "--device cuda" in command
    assert "--keep-work-dir" in command
    assert "contest_auth_eval.json" in command


def test_lossy_coarsening_harvest_requires_numeric_score_and_archive_bytes(tmp_path: Path) -> None:
    module = _load_script("experiments/lossy_coarsening_lightning_harvest.py")
    evidence_out = tmp_path / "evidence.jsonl"
    target_row = {
        "archive_bytes": 156404,
        "rel_err_budget": 0.05,
        "rel_err_actual_int8": 0.03856,
        "rel_err_actual_fp32_smoke": 0.034811,
        "predicted_band": [0.18, 0.22],
    }

    row = module._emit_evidence_row(
        evidence_out=evidence_out,
        auth_eval={"score": 0.189, "archive_bytes": 156404, "rate": 0.004},
        target_row=target_row,
        job_name="job-test",
    )

    assert row["evidence_grade"] == "[contest-CUDA]"
    assert row["score_claim"] is False
    assert row["archive_bytes_match_expected"] is True
    assert json.loads(evidence_out.read_text(encoding="utf-8"))["score_contest_cuda"] == 0.189

    with pytest.raises(SystemExit, match="without numeric auth_eval score"):
        module._emit_evidence_row(
            evidence_out=evidence_out,
            auth_eval={"archive_bytes": 156404},
            target_row=target_row,
            job_name="job-test",
        )
    with pytest.raises(SystemExit, match="archive_bytes 1 != expected 156404"):
        module._emit_evidence_row(
            evidence_out=evidence_out,
            auth_eval={"score": 0.189, "archive_bytes": 1},
            target_row=target_row,
            job_name="job-test",
        )
