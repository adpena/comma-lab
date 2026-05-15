# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from tac.deploy.kaggle.pr106_yshift_score_table import (
    DEFAULT_JOB_NAME,
    DEFAULT_SOURCE_BUNDLE_NAME,
    KagglePr106YshiftBundleSpec,
    render_launcher,
    write_bundle,
    write_source_bundle,
)
from tac.tests.tool_loader import load_repo_tool

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    return load_repo_tool(
        REPO_ROOT,
        "tools/kaggle_build_pr106_yshift_score_table.py",
        "kaggle_build_pr106_yshift_score_table_test",
    )


def _claim_ledger(path: Path, *, job_name: str = "kaggle_pr106_yshift_test") -> None:
    path.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance_job_id | predicted_eta_utc | status | notes |\n"
        "| 2026-05-11T00:00:00Z | codex:test | lane_pr106_yshift_score_table | kaggle | "
        f"{job_name} | 2026-05-11T03:00:00Z | active_dispatching | test |\n",
        encoding="utf-8",
    )


def test_render_launcher_uses_canonical_score_table_env() -> None:
    spec = KagglePr106YshiftBundleSpec(
        username="alice",
        job_name="kaggle_pr106_yshift_test",
        candidate_radius=4,
        score_step=0.5,
        batch_pairs=16,
    )

    launcher = render_launcher(spec)

    assert DEFAULT_SOURCE_BUNDLE_NAME in launcher
    assert "SOURCE_TREE_NAME = SOURCE_BUNDLE_NAME.removesuffix" in launcher
    assert "pact_pr106_yshift_workspace" in launcher
    assert "import tac.deploy.pr106_yshift" in launcher
    assert "PR106_YSHIFT_ALLOW_EXPANDED_SOURCE_TREE" in launcher
    assert "refusing expanded-source fallback" in launcher
    assert "'PR106_YSHIFT_MODE': 'score_table'" in launcher
    assert "'PR106_YSHIFT_SCORE_TABLE_INSTANCE_JOB_ID': 'kaggle_pr106_yshift_test'" in launcher
    assert "'PR106_YSHIFT_CANDIDATE_RADIUS': '4'" in launcher
    assert "'PR106_YSHIFT_SCORE_STEP': '0.5'" in launcher
    assert "'PR106_YSHIFT_SCORE_TABLE_BATCH_PAIRS': '16'" in launcher
    assert '"bash", \'scripts/remote_lane_pr106_yshift_sidechannel.sh\'' in launcher
    assert "_safe_extract_tarball(bundle, WORKSPACE)" in launcher
    assert "_copy_expanded_source_tree(source_tree, WORKSPACE)" in launcher
    assert "_ensure_pr106_archive_zip(workspace)" in launcher
    assert "torch==2.4.1+cu121" in launcher
    assert "PR106_YSHIFT_TORCH_FALLBACK_REEXEC" in launcher
    assert "PYTORCH_CUDA_ALLOC_CONF" in launcher
    assert '"PR106_YSHIFT_ALLOW_PROVIDER_CLAIM_MIRROR": "1"' in launcher
    assert "_ensure_torch_supports_visible_cuda()" in launcher
    assert "zipfile.ZipFile(archive, \"w\", compression=zipfile.ZIP_STORED)" in launcher
    compile(launcher, "run_kernel.py", "exec")


def test_write_source_bundle_contains_runtime_contract_and_claim_ledger(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    claims = tmp_path / "active_lane_dispatch_claims.md"
    _claim_ledger(claims)
    bundle = tmp_path / "bundle"
    spec = KagglePr106YshiftBundleSpec(
        username="alice",
        job_name="kaggle_pr106_yshift_test",
        dataset_ref="alice/comma-lab-private-assets",
        source_dataset_ref="alice/comma-lab-pr106-yshift-source",
    )

    manifest = write_source_bundle(
        repo_root=REPO_ROOT,
        output_path=bundle / DEFAULT_SOURCE_BUNDLE_NAME,
        spec=spec,
        pr106_archive=archive,
        claims_path=claims,
    )

    assert manifest["schema"] == "kaggle_pr106_yshift_source_bundle_v1"
    with tarfile.open(bundle / DEFAULT_SOURCE_BUNDLE_NAME, "r:gz") as tar:
        names = set(tar.getnames())

    assert "inputs/pr106_archive.zip" in names
    assert ".omx/state/active_lane_dispatch_claims.md" in names
    assert "src/tac/deploy/pr106_yshift.py" in names
    assert "experiments/build_pr106_yshift_score_table.py" in names
    assert "scripts/remote_lane_pr106_yshift_sidechannel.sh" in names
    assert "submissions/pr106_yshift_sidechannel/inflate.py" in names


def test_write_source_bundle_is_byte_reproducible_across_output_paths(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    claims = tmp_path / "active_lane_dispatch_claims.md"
    _claim_ledger(claims)
    spec = KagglePr106YshiftBundleSpec(
        username="alice",
        job_name="kaggle_pr106_yshift_test",
        dataset_ref="alice/comma-lab-private-assets",
        source_dataset_ref="alice/comma-lab-pr106-yshift-source",
    )
    first = tmp_path / "a" / DEFAULT_SOURCE_BUNDLE_NAME
    second = tmp_path / "b" / "renamed-yshift-source.tar.gz"

    write_source_bundle(
        repo_root=REPO_ROOT,
        output_path=first,
        spec=spec,
        pr106_archive=archive,
        claims_path=claims,
    )
    write_source_bundle(
        repo_root=REPO_ROOT,
        output_path=second,
        spec=spec,
        pr106_archive=archive,
        claims_path=claims,
    )

    assert first.read_bytes() == second.read_bytes()


def test_write_bundle_declares_source_dataset_and_inlines_fresh_source_bundle(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    claims = tmp_path / "active_lane_dispatch_claims.md"
    _claim_ledger(claims)
    bundle = tmp_path / "bundle"
    spec = KagglePr106YshiftBundleSpec(
        username="alice",
        job_name="kaggle_pr106_yshift_test",
        dataset_ref="alice/comma-lab-private-assets",
        source_dataset_ref="alice/comma-lab-pr106-yshift-source",
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
    assert metadata["dataset_sources"] == [
        "alice/comma-lab-pr106-yshift-source",
        "alice/comma-lab-private-assets",
    ]
    assert metadata["launch_policy"]["score_claim"] is False
    assert manifest["score_claim"] is False
    assert manifest["schema"] == "kaggle_pr106_yshift_score_table_bundle_v2"
    assert manifest["inline_source_bundle"] == DEFAULT_SOURCE_BUNDLE_NAME
    assert (bundle / DEFAULT_SOURCE_BUNDLE_NAME).is_file()
    assert not (bundle / "inputs/pr106_archive.zip").exists()
    assert not (bundle / ".omx/state/active_lane_dispatch_claims.md").exists()
    assert not (bundle / "src/tac/deploy/pr106_yshift.py").exists()
    assert "score_claim" in launcher

    with tarfile.open(bundle / DEFAULT_SOURCE_BUNDLE_NAME, "r:gz") as tar:
        names = set(tar.getnames())
    assert ".omx/state/active_lane_dispatch_claims.md" in names
    assert "inputs/pr106_archive.zip" in names


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
    _claim_ledger(claims, job_name="wrong_job")
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
