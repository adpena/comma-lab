from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tac.deploy.kaggle.pr106_yshift_score_table import (
    DEFAULT_JOB_NAME,
    KagglePr106YshiftBundleSpec,
    render_launcher,
    write_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "kaggle_build_pr106_yshift_score_table.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "kaggle_build_pr106_yshift_score_table_test",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_render_launcher_uses_canonical_score_table_env() -> None:
    spec = KagglePr106YshiftBundleSpec(
        username="alice",
        job_name="kaggle_pr106_yshift_test",
        candidate_radius=4,
        score_step=0.5,
        batch_pairs=16,
    )

    launcher = render_launcher(spec)

    assert "verify_submodule=\"tac.deploy.pr106_yshift\"" in launcher
    assert "'PR106_YSHIFT_MODE': 'score_table'" in launcher
    assert "'PR106_YSHIFT_SCORE_TABLE_INSTANCE_JOB_ID': 'kaggle_pr106_yshift_test'" in launcher
    assert "'PR106_YSHIFT_CANDIDATE_RADIUS': '4'" in launcher
    assert "'PR106_YSHIFT_SCORE_STEP': '0.5'" in launcher
    assert "'PR106_YSHIFT_SCORE_TABLE_BATCH_PAIRS': '16'" in launcher
    assert '"bash", \'scripts/remote_lane_pr106_yshift_sidechannel.sh\'' in launcher
    compile(launcher, "run_kernel.py", "exec")


def test_write_bundle_copies_runtime_contract_and_claim_ledger(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    claims = tmp_path / "active_lane_dispatch_claims.md"
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance_job_id | predicted_eta_utc | status | notes |\n"
        "| 2026-05-11T00:00:00Z | codex:test | lane_pr106_yshift_score_table | kaggle | "
        "kaggle_pr106_yshift_test | 2026-05-11T03:00:00Z | active_dispatching | test |\n",
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle"
    spec = KagglePr106YshiftBundleSpec(
        username="alice",
        job_name="kaggle_pr106_yshift_test",
        dataset_ref="alice/comma-lab-private-assets",
    )

    manifest = write_bundle(
        repo_root=REPO_ROOT,
        bundle_dir=bundle,
        spec=spec,
        pr106_archive=archive,
        claims_path=claims,
    )

    metadata = json.loads((bundle / "kernel-metadata.json").read_text(encoding="utf-8"))
    launcher = (bundle / "run_kernel.py").read_text(encoding="utf-8")

    assert metadata["id"] == "alice/comma-lab-pr106-yshift-score-table"
    assert metadata["enable_gpu"] is True
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == ["alice/comma-lab-private-assets"]
    assert metadata["launch_policy"]["score_claim"] is False
    assert manifest["score_claim"] is False
    assert (bundle / "inputs/pr106_archive.zip").read_bytes() == b"archive"
    assert (bundle / ".omx/state/active_lane_dispatch_claims.md").is_file()
    assert (bundle / "src/tac/deploy/pr106_yshift.py").is_file()
    assert (bundle / "experiments/build_pr106_yshift_score_table.py").is_file()
    assert (bundle / "scripts/remote_lane_pr106_yshift_sidechannel.sh").is_file()
    assert (bundle / "submissions/pr106_yshift_sidechannel/inflate.py").is_file()
    assert "score_claim" in launcher


def test_write_bundle_requires_claim_ledger(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    spec = KagglePr106YshiftBundleSpec(username="alice")

    with pytest.raises(FileNotFoundError, match="claim"):
        write_bundle(
            repo_root=REPO_ROOT,
            bundle_dir=tmp_path / "bundle",
            spec=spec,
            pr106_archive=archive,
            claims_path=tmp_path / "missing.md",
        )


def test_write_bundle_requires_matching_active_claim(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    claims = tmp_path / "active_lane_dispatch_claims.md"
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance_job_id | predicted_eta_utc | status | notes |\n"
        "| 2026-05-11T00:00:00Z | codex:test | lane_pr106_yshift_score_table | kaggle | "
        "wrong_job | 2026-05-11T03:00:00Z | active_dispatching | test |\n",
        encoding="utf-8",
    )
    spec = KagglePr106YshiftBundleSpec(
        username="alice",
        job_name="kaggle_pr106_yshift_test",
    )

    with pytest.raises(ValueError, match="no active lane claim"):
        write_bundle(
            repo_root=REPO_ROOT,
            bundle_dir=tmp_path / "bundle",
            spec=spec,
            pr106_archive=archive,
            claims_path=claims,
        )


def test_tool_print_claim_uses_kaggle_platform(capsys: pytest.CaptureFixture[str]) -> None:
    tool = _load_tool()

    assert tool.main(["--username", "alice", "--print-claim"]) == 0
    output = capsys.readouterr().out

    assert "--lane-id \\\n+  lane_pr106_yshift_score_table" in output
    assert "--platform \\\n+  kaggle" in output
    assert f"--instance-job-id \\\n+  {DEFAULT_JOB_NAME}" in output
