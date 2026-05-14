# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "harvest_pr106_yshift_score_table_batch.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "harvest_pr106_yshift_score_table_batch_test", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_mirror_accepts_cuda_score_table_artifacts(tmp_path: Path) -> None:
    tool = _load_tool()
    mirror = tmp_path / "mirror"
    (mirror / "yshift_run/build").mkdir(parents=True)
    (mirror / "yshift_run/score_table").mkdir(parents=True)
    (mirror / "lightning_runner_preflight.json").write_text(
        json.dumps(
            {
                "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK": True,
                "gpu_names": ["Tesla T4"],
                "torch_version": "2.5.1+cu124",
                "torch_cuda": "12.4",
            }
        )
    )
    (mirror / "pr106_yshift_score_table_batch_summary.json").write_text(
        json.dumps({"contest_auth_eval_json_exists": True})
    )
    (mirror / "contest_auth_eval.json").write_text(json.dumps({"final_score": 0.207}))
    (mirror / "batch_run.log").write_text("ok\n")
    (mirror / "yshift_run/build/pr106_yshift_sidechannel_archive.zip").write_bytes(b"zip")
    (mirror / "yshift_run/score_table/score_table_manifest.json").write_text(
        json.dumps({"score_claim": False})
    )

    validation = tool.validate_mirror(mirror)

    assert validation["status"] == "validated"
    assert validation["evidence_grade"] == "A_pending_adjudication"
    assert validation["final_score"] == 0.207
    assert (mirror / "pr106_yshift_batch_harvest_validation.json").is_file()


def test_running_job_missing_final_artifacts_is_not_terminal_claim(
    tmp_path: Path, monkeypatch
) -> None:
    tool = _load_tool()
    state = tmp_path / "lightning_batch_jobs.json"
    state.write_text(
        json.dumps(
            [
                {
                    "status": "Running",
                    "spec": {
                        "name": "lane_pr106_yshift_score_table_test",
                        "remote_output_dir": "/remote/out",
                        "local_artifact_dir": "experiments/results/missing_yshift_test",
                    },
                    "job": {"status": "Running"},
                }
            ]
        ),
        encoding="utf-8",
    )
    close_calls: list[dict[str, object]] = []

    def fake_validate(_mirror: Path) -> dict[str, object]:
        raise FileNotFoundError("missing PR106 yshift batch artifacts: ['contest_auth_eval.json']")

    def fake_close_claim(*_args, **kwargs) -> None:
        close_calls.append(kwargs)

    monkeypatch.setattr(tool, "validate_mirror", fake_validate)
    monkeypatch.setattr(tool, "close_claim", fake_close_claim)

    try:
        tool.main(
            [
                "--job-name",
                "lane_pr106_yshift_score_table_test",
                "--state-path",
                str(state),
                "--no-copy",
                "--close-claim",
            ]
        )
    except SystemExit as exc:
        assert str(exc).startswith("ARTIFACT_NOT_READY:")
    else:  # pragma: no cover
        raise AssertionError("expected ARTIFACT_NOT_READY SystemExit")

    assert close_calls == []
