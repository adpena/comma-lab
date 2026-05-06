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
