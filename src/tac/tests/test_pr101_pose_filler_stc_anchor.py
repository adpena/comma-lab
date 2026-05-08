from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "pr101_pose_filler_stc_anchor.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pr101_pose_filler_stc_anchor", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pose_filler_stc_anchor_fallback_run_is_non_promotable(tmp_path: Path) -> None:
    tool = _load_tool()
    out_dir = tmp_path / "fstc_anchor"

    rc = tool.main(["--poses", str(tmp_path / "missing.pt"), "--output-dir", str(out_dir)])

    manifest = json.loads((out_dir / "build_manifest.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert (out_dir / "fstc_pose_blob.bin").is_file()
    assert manifest["candidate_id"] == "pose_codec/filler_stc_v1"
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "substrate_pr101_monolithic_no_separate_pose_payload" in manifest["dispatch_blockers"]
    assert manifest["fstc_blob_bytes"] > 0
    assert manifest["pd_v2_blob_bytes"] > 0
