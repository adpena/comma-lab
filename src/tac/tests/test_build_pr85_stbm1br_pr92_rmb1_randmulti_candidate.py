from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_pr85_stbm1br_pr92_rmb1_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _require_real_inputs() -> None:
    required = [
        module.DEFAULT_PR85_ARCHIVE,
        module.DEFAULT_STBM_ARCHIVE,
        module.DEFAULT_STBM_MANIFEST,
        module.DEFAULT_PR92_ARCHIVE,
        module.DEFAULT_PR92_PROFILE,
        module.DEFAULT_STBM_REPLAY_RUNTIME / "inflate.py",
        module.DEFAULT_STBM_EXACT_T4,
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        pytest.skip("required local PR85/STBM/PR92 artifacts are missing: " + ", ".join(missing))


def test_real_pr85_stbm1br_pr92_rmb1_candidate_builds_with_dispatch_guards(tmp_path: Path) -> None:
    _require_real_inputs()

    summary = module.build_candidate(out_dir=tmp_path / "candidate")
    manifest_path = REPO / summary["candidate_manifest"] if not Path(summary["candidate_manifest"]).is_absolute() else Path(summary["candidate_manifest"])
    if not manifest_path.is_file():
        manifest_path = tmp_path / "candidate" / module.CANDIDATE_ID / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary["score_claim"] is False
    assert summary["dispatch_performed"] is False
    assert summary["archive_delta_bytes_vs_stbm"] == -276
    assert summary["strict_zip_valid"] is True
    assert summary["exact_t4_dispatch_justified_after_claim"] is True
    assert manifest["candidate_archive"]["archive_bytes"] == 229480
    assert manifest["candidate_archive"]["archive_sha256"]
    assert manifest["non_noop_byte_change"]["randmulti_delta_bytes_vs_stbm"] == -276
    assert manifest["stbm_mask_preservation"]["unchanged"] is True
    assert manifest["randmulti_decoded_row_parity"]["decoded_rows_match"] is True
    assert manifest["randmulti_decoded_row_parity"]["decoded_rows_sha256"] == module.EXPECTED["decoded_randmulti_rows_sha256"]
    assert [row["segment"] for row in manifest["segment_diffs"]["candidate_vs_stbm"]] == ["randmulti"]
    assert manifest["runtime_support"]["candidate_replay_runtime"]["status"] == "passed"
    assert manifest["runtime_support"]["robust_current_apply_qzs3_rmb1"]["status"] == "passed"
    assert manifest["exact_eval_runtime_contract"]["ready_for_exact_eval_runtime"] is True
    assert manifest["exact_eval_runtime_contract"]["runtime_tree_sha256"]
    assert manifest["exact_eval_runtime_contract"]["remaining_blockers"] == []
    assert manifest["dispatch_readiness"]["checks"]["candidate_score_claim_false"] is True
    assert manifest["dispatch_readiness"]["checks"]["remote_dispatch_not_performed"] is True
    assert "tools/claim_lane_dispatch.py claim" in manifest["dispatch_readiness"]["next_claim_command"]

    archive_path = REPO / manifest["candidate_archive"]["path"]
    if not archive_path.is_file():
        archive_path = tmp_path / "candidate" / module.CANDIDATE_ID / "archive.zip"
    with zipfile.ZipFile(archive_path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        assert [info.filename for info in infos] == ["x"]
        assert infos[0].compress_type == zipfile.ZIP_STORED
