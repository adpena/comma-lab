# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from tac.deploy.kaggle.pr106_latent_score_table import (
    DEFAULT_JOB_NAME,
    DEFAULT_SOURCE_BUNDLE_NAME,
    PINNED_UPSTREAM_COMMIT,
    KagglePr106LatentBundleSpec,
    render_launcher,
    write_bundle,
    write_source_bundle,
)
from tac.tests.tool_loader import load_repo_tool

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    return load_repo_tool(
        REPO_ROOT,
        "tools/kaggle_build_pr106_latent_score_table.py",
        "kaggle_build_pr106_latent_score_table_test",
    )


def _claim_ledger(path: Path, *, job_name: str = "kaggle_pr106_latent_test") -> None:
    path.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance_job_id | predicted_eta_utc | status | notes |\n"
        "| 2026-05-11T00:00:00Z | codex:test | lane_pr106_latent_sidecar | kaggle | "
        f"{job_name} | 2026-05-11T03:00:00Z | active_dispatching | test |\n",
        encoding="utf-8",
    )


def test_render_launcher_uses_canonical_latent_score_table_env() -> None:
    spec = KagglePr106LatentBundleSpec(
        username="alice",
        job_name="kaggle_pr106_latent_test",
        delta_radius=2,
        batch_pairs=3,
        candidate_batch_size=5,
        sidecar_top_k=400,
    )

    launcher = render_launcher(spec)

    assert DEFAULT_SOURCE_BUNDLE_NAME in launcher
    assert "pact_pr106_latent_workspace" in launcher
    assert "import tac.deploy.pr106_latent" in launcher
    assert "PR106_LATENT_ALLOW_EXPANDED_SOURCE_TREE" in launcher
    assert "_find_verified_expanded_source_bundle_tree" in launcher
    assert "verified expanded " in launcher
    assert "source-bundle tree" in launcher
    assert "refusing expanded-source fallback" in launcher
    assert "'PR106_LATENT_MODE': 'score_table'" in launcher
    assert "'PR106_LATENT_DELTA_RADIUS': '2'" in launcher
    assert "'PR106_ARCHIVE_MEMBER': 'x'" in launcher
    assert "'PR106_EXPECTED_ARCHIVE_SHA256': '56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7'" in launcher
    assert "'PR106_EXPECTED_ARCHIVE_MEMBER_SHA256': '852a4cb1231413cf1a8fc867e2a808de9ec78511d2ebf283df2c5b608cb4a749'" in launcher
    assert "'PR106_RUNTIME_DIR': 'submissions/pr106_latent_sidecar_r2_pr101_grammar'" in launcher
    assert "'PR106_LATENT_SCORE_TABLE_BATCH_PAIRS': '3'" in launcher
    assert "'PR106_LATENT_SCORE_TABLE_CANDIDATE_BATCH_SIZE': '5'" in launcher
    assert "'PR106_LATENT_SCORE_TABLE_INSTANCE_JOB_ID': 'kaggle_pr106_latent_test'" in launcher
    assert "'SIDECAR_TOP_K': '400'" in launcher
    assert '"bash", \'scripts/remote_lane_pr106_latent_sidecar.sh\'' in launcher
    assert "torch==2.4.1+cu121" in launcher
    assert "constriction==0.4.2" in launcher
    assert "segmentation-models-pytorch==0.5.0" in launcher
    assert f"UPSTREAM_COMMIT = '{PINNED_UPSTREAM_COMMIT}'" in launcher
    assert '"git", "fetch", "--depth", "1", "origin", UPSTREAM_COMMIT' in launcher
    assert '"TAC_UPSTREAM_COMMIT": upstream_commit' in launcher
    assert "PR106_LATENT_TORCH_FALLBACK_REEXEC" in launcher
    assert "PYTORCH_CUDA_ALLOC_CONF" in launcher
    assert '"PR106_LATENT_ALLOW_PROVIDER_CLAIM_MIRROR": "1"' in launcher
    assert "archive_member = 'x'" in launcher
    assert 'zipfile.ZipInfo(archive_member' in launcher
    compile(launcher, "run_kernel.py", "exec")


def test_launcher_accepts_verified_expanded_latent_source_bundle_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"payload"
    payload_sha = hashlib.sha256(payload).hexdigest()
    spec = KagglePr106LatentBundleSpec(
        username="alice",
        job_name="kaggle_pr106_latent_test",
        expected_archive_sha256="a" * 64,
        expected_archive_member_sha256=payload_sha,
    )
    launcher = render_launcher(spec)
    script = tmp_path / "script.py"
    script.write_text(launcher, encoding="utf-8")
    namespace = {"__file__": str(script), "__name__": "generated_launcher"}
    exec(compile(launcher, str(script), "exec"), namespace)

    input_root = tmp_path / "input"
    tree = input_root / DEFAULT_SOURCE_BUNDLE_NAME.removesuffix(".tar.gz")
    for rel in (
        "src/tac",
        ".omx/state",
        "experiments",
        "scripts",
        "submissions/pr106_latent_sidecar_r2_pr101_grammar",
        "inputs/pr106_archive",
    ):
        (tree / rel).mkdir(parents=True, exist_ok=True)
    (tree / ".omx/state/active_lane_dispatch_claims.md").write_text("claims\n", encoding="utf-8")
    (tree / "experiments/build_pr106_latent_score_table.py").write_text("pass\n", encoding="utf-8")
    (tree / "scripts/remote_lane_pr106_latent_sidecar.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (tree / "source_bundle_manifest.json").write_text(
        json.dumps(
            {
                "schema": "kaggle_pr106_latent_source_bundle_v1",
                "job_name": "kaggle_pr106_latent_test",
                "lane_id": "lane_pr106_latent_sidecar",
                "archive_member": "x",
                "expected_archive_sha256": "a" * 64,
                "expected_archive_member_sha256": payload_sha,
                "runtime_dir": "submissions/pr106_latent_sidecar_r2_pr101_grammar",
                "delta_radius": 2,
                "upstream_commit": PINNED_UPSTREAM_COMMIT,
                "score_claim": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tree / "inputs/pr106_archive/x").write_bytes(payload)

    monkeypatch.setenv("CLOUD_INPUT_ROOT", str(input_root))

    assert namespace["_find_verified_expanded_source_bundle_tree"]() == tree


def test_launcher_rejects_arbitrary_expanded_latent_source_without_local_dev_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = KagglePr106LatentBundleSpec(username="alice", job_name="kaggle_pr106_latent_test")
    launcher = render_launcher(spec)
    script = tmp_path / "script.py"
    script.write_text(launcher, encoding="utf-8")
    namespace = {"__file__": str(script), "__name__": "generated_launcher"}
    exec(compile(launcher, str(script), "exec"), namespace)
    arbitrary = tmp_path / "input" / "some_repo" / "src" / "tac"
    arbitrary.mkdir(parents=True)
    monkeypatch.setenv("CLOUD_INPUT_ROOT", str(tmp_path / "input"))
    monkeypatch.delenv("PR106_LATENT_ALLOW_EXPANDED_SOURCE_TREE", raising=False)

    assert namespace["_find_verified_expanded_source_bundle_tree"]() is None
    assert namespace["_find_expanded_source_tree"]() == arbitrary.parent.parent


def test_write_source_bundle_contains_latent_runtime_contract_and_claim_ledger(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    claims = tmp_path / "active_lane_dispatch_claims.md"
    _claim_ledger(claims)
    bundle = tmp_path / "bundle"
    spec = KagglePr106LatentBundleSpec(
        username="alice",
        job_name="kaggle_pr106_latent_test",
        dataset_ref="alice/comma-lab-private-assets",
        source_dataset_ref="alice/comma-lab-pr106-latent-source",
    )

    manifest = write_source_bundle(
        repo_root=REPO_ROOT,
        output_path=bundle / DEFAULT_SOURCE_BUNDLE_NAME,
        spec=spec,
        pr106_archive=archive,
        claims_path=claims,
    )

    assert manifest["schema"] == "kaggle_pr106_latent_source_bundle_v1"
    assert manifest["archive_member"] == "x"
    assert manifest["expected_archive_sha256"] == "56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7"
    assert manifest["expected_archive_member_sha256"] == "852a4cb1231413cf1a8fc867e2a808de9ec78511d2ebf283df2c5b608cb4a749"
    assert manifest["runtime_dir"] == "submissions/pr106_latent_sidecar_r2_pr101_grammar"
    assert manifest["delta_radius"] == 2
    assert manifest["upstream_commit"] == PINNED_UPSTREAM_COMMIT
    with tarfile.open(bundle / DEFAULT_SOURCE_BUNDLE_NAME, "r:gz") as tar:
        names = set(tar.getnames())

    assert "source_bundle_manifest.json" in names
    assert "inputs/pr106_archive.zip" in names
    assert ".omx/state/active_lane_dispatch_claims.md" in names
    assert "src/tac/deploy/pr106_latent.py" in names
    assert "experiments/build_pr106_latent_score_table.py" in names
    assert "scripts/remote_lane_pr106_latent_sidecar.sh" in names
    assert "submissions/pr106_latent_sidecar/inflate.py" in names
    assert "tools/materialize_pr106_latent_score_table_candidate.py" in names


def test_latent_source_bundle_contains_remote_script_tool_closure(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    claims = tmp_path / "active_lane_dispatch_claims.md"
    _claim_ledger(claims)
    bundle = tmp_path / "bundle"
    spec = KagglePr106LatentBundleSpec(username="alice", job_name="kaggle_pr106_latent_test")

    write_source_bundle(
        repo_root=REPO_ROOT,
        output_path=bundle / DEFAULT_SOURCE_BUNDLE_NAME,
        spec=spec,
        pr106_archive=archive,
        claims_path=claims,
    )

    with tarfile.open(bundle / DEFAULT_SOURCE_BUNDLE_NAME, "r:gz") as tar:
        names = set(tar.getnames())

    assert "source_bundle_manifest.json" in names
    remote_script = (REPO_ROOT / "scripts/remote_lane_pr106_latent_sidecar.sh").read_text(
        encoding="utf-8"
    )
    invoked_tools = {
        "tools/materialize_pr106_latent_score_table_candidate.py",
        "tools/prove_pr106_sidecar_runtime_consumption.py",
    }
    for rel in invoked_tools:
        assert rel in remote_script
        assert rel in names


def test_write_source_bundle_is_byte_reproducible_across_output_paths(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    claims = tmp_path / "active_lane_dispatch_claims.md"
    _claim_ledger(claims)
    spec = KagglePr106LatentBundleSpec(
        username="alice",
        job_name="kaggle_pr106_latent_test",
        dataset_ref="alice/comma-lab-private-assets",
        source_dataset_ref="alice/comma-lab-pr106-latent-source",
    )
    first = tmp_path / "a" / DEFAULT_SOURCE_BUNDLE_NAME
    second = tmp_path / "b" / "renamed-latent-source.tar.gz"

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


def test_write_bundle_declares_latent_source_dataset_and_inlines_fresh_source_bundle(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    claims = tmp_path / "active_lane_dispatch_claims.md"
    _claim_ledger(claims)
    bundle = tmp_path / "bundle"
    spec = KagglePr106LatentBundleSpec(
        username="alice",
        job_name="kaggle_pr106_latent_test",
        dataset_ref="alice/comma-lab-private-assets",
        source_dataset_ref="alice/comma-lab-pr106-latent-source",
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

    assert metadata["id"] == "alice/comma-lab-pr106-latent-score-table"
    assert metadata["enable_gpu"] is True
    assert metadata["dataset_sources"] == [
        "alice/comma-lab-pr106-latent-source",
        "alice/comma-lab-private-assets",
    ]
    assert metadata["launch_policy"]["score_claim"] is False
    assert manifest["schema"] == "kaggle_pr106_latent_score_table_bundle_v1"
    assert manifest["score_claim"] is False
    assert manifest["inline_source_bundle"] == DEFAULT_SOURCE_BUNDLE_NAME
    assert manifest["upstream_commit"] == PINNED_UPSTREAM_COMMIT
    assert (bundle / DEFAULT_SOURCE_BUNDLE_NAME).is_file()
    assert not (bundle / "inputs/pr106_archive.zip").exists()
    assert "score_claim" in launcher

    with tarfile.open(bundle / DEFAULT_SOURCE_BUNDLE_NAME, "r:gz") as tar:
        names = set(tar.getnames())
    assert ".omx/state/active_lane_dispatch_claims.md" in names
    assert "inputs/pr106_archive.zip" in names


def test_write_bundle_requires_matching_active_latent_claim(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    claims = tmp_path / "active_lane_dispatch_claims.md"
    _claim_ledger(claims, job_name="wrong_job")
    spec = KagglePr106LatentBundleSpec(
        username="alice",
        job_name="kaggle_pr106_latent_test",
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

    assert "--lane-id \\\n+  lane_pr106_latent_sidecar" in output
    assert "--platform \\\n+  kaggle" in output
    assert f"--instance-job-id \\\n+  {DEFAULT_JOB_NAME}" in output
