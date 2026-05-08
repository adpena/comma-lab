from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import stat
import sys
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "build_pr102_exact_replay_readiness.py"


def _load_module():
    for path in (REPO, REPO / "tools"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    spec = importlib.util.spec_from_file_location("build_pr102_exact_replay_readiness_under_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_module()


def test_pr102_readiness_records_custody_risks_and_command(tmp_path: Path) -> None:
    fixture = _write_pr102_fixture(tmp_path)

    report = tool.build_pr102_exact_replay_readiness(
        manifest_path=fixture["manifest"],
        repo_root=tmp_path,
        adapter_rel_path="experiments/results/pr102_adapter/inflate.sh",
    )

    payload = report.to_dict()
    assert payload["adapter_plan_ready"] is True
    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["summary"]["archive"]["sha256"] == fixture["archive_sha256"]
    assert payload["summary"]["archive"]["members"][0]["sha256"] == fixture["member_sha256"]
    assert payload["summary"]["archive"]["clean_clone_restore_required"] is True
    assert payload["summary"]["runtime_source"]["file_count"] == 7
    assert payload["summary"]["runtime_source"]["source_tree_sha256"]

    risks = payload["summary"]["dependency_network_risks"]
    assert risks["required_python_modules_for_adapter_preflight"] == ["brotli", "numpy", "torch"]
    assert any("pip install" in finding["text"] for finding in risks["network_or_install_findings"])
    assert any("pip', 'install" in finding["text"] for finding in risks["network_or_install_findings"])
    assert any("curl" in finding["text"] for finding in risks["network_or_install_findings"])

    adapter = payload["summary"]["adapter_plan"]
    assert adapter["contest_auth_eval_command"] == [
        ".venv/bin/python",
        "experiments/contest_auth_eval.py",
        "--archive",
        "experiments/results/pr102/archive.zip",
        "--inflate-sh",
        "experiments/results/pr102_adapter/inflate.sh",
        "--upstream-dir",
        "upstream",
        "--device",
        "cuda",
    ]
    assert "pip install" not in adapter["inflate_sh_text"]
    assert "PACT_RUNTIME_DEPENDENCY_ROOT" in adapter["inflate_sh_text"]
    assert 'if [ "$#" -ne 3 ]; then' in adapter["inflate_sh_text"]
    assert "required PR102 runtime dependency missing" in adapter["inflate_sh_text"]
    assert 'export PYTHONPATH="$PUBLIC_SOURCE_ROOT:$RUNTIME_SOURCE_ROOT"' in adapter["inflate_sh_text"]
    assert "PYTHONPATH:-" not in adapter["inflate_sh_text"]

    runbook = payload["summary"]["lightning_exact_eval_runbook"]
    assert runbook["lane_id"] == "pr102_public_exact_replay_t4"
    assert runbook["job_name"] == "pr102-hnerv-lc-v2-scale095-rplus1-exact"
    assert runbook["claim_command"][:3] == [
        ".venv/bin/python",
        "tools/claim_lane_dispatch.py",
        "claim",
    ]
    assert "--stage-workspace" in runbook["wrapper_submit_command"]
    assert "--require-stage-cuda" not in runbook["wrapper_submit_command"]
    assert "${LIGHTNING_MACHINE:-g4dn.2xlarge}" in runbook["wrapper_submit_command"]
    assert "--source-manifest" not in runbook["wrapper_submit_command"]
    assert "--submit" in runbook["wrapper_submit_command"]
    assert "INFLATE_TORCHVISION_SPEC=torchvision==0.20.1+cu124" in runbook["wrapper_submit_command"]
    assert "experiments/results/pr102/archive.zip" in runbook["source_manifest_expected_artifacts"]
    assert "experiments/results/pr102_adapter/inflate.sh" in runbook["source_manifest_expected_artifacts"]
    assert "experiments/results/pr102_adapter/readiness.json" in runbook["source_manifest_expected_artifacts"]
    assert "experiments/results/pr102_adapter/readiness.md" in runbook["source_manifest_expected_artifacts"]
    assert any(
        "source_manifest" in guardrail
        for guardrail in runbook["guardrails"]
    )
    assert any(
        "restore the external archive artifact" in guardrail
        for guardrail in runbook["guardrails"]
    )


def test_pr102_readiness_fails_closed_on_archive_hash_mismatch(tmp_path: Path) -> None:
    fixture = _write_pr102_fixture(tmp_path)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    manifest["entries"][0]["archive"]["sha256"] = "0" * 64
    fixture["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = tool.build_pr102_exact_replay_readiness(
        manifest_path=fixture["manifest"],
        repo_root=tmp_path,
    )

    assert report.ready is False
    assert "archive_sha256_mismatch" in report.blockers
    assert report.to_dict()["summary"]["adapter_plan"]["contest_auth_eval_command"] == []


def test_materialize_adapter_plan_writes_source_sized_files(tmp_path: Path) -> None:
    fixture = _write_pr102_fixture(tmp_path)
    report = tool.build_pr102_exact_replay_readiness(
        manifest_path=fixture["manifest"],
        repo_root=tmp_path,
        adapter_rel_path="experiments/results/pr102_adapter/inflate.sh",
    )

    materialized = tool.materialize_adapter_plan(
        report,
        adapter_dir=tmp_path / "experiments/results/pr102_adapter",
        repo_root=tmp_path,
    )

    inflate_sh = tmp_path / "experiments/results/pr102_adapter/inflate.sh"
    readme = tmp_path / "experiments/results/pr102_adapter/README.md"
    assert inflate_sh.is_file()
    assert readme.is_file()
    assert inflate_sh.stat().st_mode & stat.S_IXUSR
    inflate_text = inflate_sh.read_text(encoding="utf-8")
    assert "pip install" not in inflate_text
    assert "experiments/results/pr102_adapter/runtime_source" in inflate_text
    assert 'export PYTHONPATH="$PUBLIC_SOURCE_ROOT:$RUNTIME_SOURCE_ROOT"' in inflate_text
    assert "PYTHONPATH:-" not in inflate_text
    runtime_inflate = (
        tmp_path
        / "experiments/results/pr102_adapter/runtime_source/submissions/"
        "hnerv_lc_v2_scale095_rplus1/inflate.py"
    )
    assert runtime_inflate.is_file()
    materialized_files = materialized.summary["adapter_plan"]["materialized_files"]
    assert materialized_files[:2] == [
        "experiments/results/pr102_adapter/inflate.sh",
        "experiments/results/pr102_adapter/README.md",
    ]
    assert (
        "experiments/results/pr102_adapter/runtime_source/submissions/"
        "hnerv_lc_v2_scale095_rplus1/inflate.py"
    ) in materialized_files
    assert (
        materialized.summary["next_status_after_this_artifact"]
        == "adapter_materialized_ready_for_exact_cuda_replay_after_dispatch_claim"
    )
    source_manifest_artifacts = materialized.summary["lightning_exact_eval_runbook"][
        "source_manifest_expected_artifacts"
    ]
    assert "experiments/results/pr102/archive.zip" in source_manifest_artifacts
    for path in materialized_files:
        assert path in source_manifest_artifacts


def test_readiness_uses_materialized_runtime_fallback_when_manifest_source_is_missing(
    tmp_path: Path,
) -> None:
    fixture = _write_pr102_fixture(tmp_path)
    adapter_rel_path = "experiments/results/pr102_adapter/inflate.sh"
    report = tool.build_pr102_exact_replay_readiness(
        manifest_path=fixture["manifest"],
        repo_root=tmp_path,
        adapter_rel_path=adapter_rel_path,
    )
    tool.materialize_adapter_plan(
        report,
        adapter_dir=tmp_path / "experiments/results/pr102_adapter",
        repo_root=tmp_path,
    )

    shutil.rmtree(tmp_path / "experiments/results/pr102/source")

    fallback_report = tool.build_pr102_exact_replay_readiness(
        manifest_path=fixture["manifest"],
        repo_root=tmp_path,
        adapter_rel_path=adapter_rel_path,
    )

    payload = fallback_report.to_dict()
    assert fallback_report.ready is True
    assert "runtime_source_root_missing" not in payload["blockers"]
    runtime = payload["summary"]["runtime_source"]
    assert runtime["source_resolution"] == "materialized_adapter_runtime_fallback"
    assert runtime["manifest_source_root"] == (
        "experiments/results/pr102/source/submissions/hnerv_lc_v2_scale095_rplus1"
    )
    assert runtime["source_root"] == (
        "experiments/results/pr102_adapter/runtime_source/"
        "submissions/hnerv_lc_v2_scale095_rplus1"
    )
    assert runtime["public_checkout_root"] == "experiments/results/pr102_adapter/runtime_source"


def test_render_markdown_includes_lightning_source_manifest_runbook(tmp_path: Path) -> None:
    fixture = _write_pr102_fixture(tmp_path)
    report = tool.build_pr102_exact_replay_readiness(
        manifest_path=fixture["manifest"],
        repo_root=tmp_path,
        adapter_rel_path="experiments/results/pr102_adapter/inflate.sh",
    )

    markdown = tool.render_markdown(report)

    assert "## Lightning Exact Eval Source Manifest Runbook" in markdown
    assert "tools/claim_lane_dispatch.py claim" in markdown
    assert "scripts/lightning_exact_eval_repro.py" in markdown
    assert "--dispatch-lane-id pr102_public_exact_replay_t4" in markdown
    assert "g4dn.2xlarge" in markdown
    assert "--require-stage-cuda" not in markdown
    assert "source_manifest.json" in markdown


def test_materialize_adapter_refuses_omx_state(tmp_path: Path) -> None:
    fixture = _write_pr102_fixture(tmp_path)
    report = tool.build_pr102_exact_replay_readiness(
        manifest_path=fixture["manifest"],
        repo_root=tmp_path,
    )
    (tmp_path / ".omx/state").mkdir(parents=True)

    with pytest.raises(ValueError, match=r"\.omx/state"):
        tool.materialize_adapter_plan(
            report,
            adapter_dir=tmp_path / ".omx/state/pr102_adapter",
            repo_root=tmp_path,
        )


def _write_pr102_fixture(repo_root: Path) -> dict[str, object]:
    (repo_root / "experiments/contest_auth_eval.py").parent.mkdir(parents=True)
    (repo_root / "experiments/contest_auth_eval.py").write_text("# contest auth eval placeholder\n")
    archive_path = repo_root / "experiments/results/pr102/archive.zip"
    member = _write_archive(archive_path, b"pr102-payload")
    runtime_root = repo_root / "experiments/results/pr102/source/submissions/hnerv_lc_v2_scale095_rplus1"
    runtime_files = {
        "README.md": "PR102 fixture runtime\n",
        "compress.sh": "#!/usr/bin/env bash\ncurl -L https://example.invalid/archive.zip -o archive.zip\n",
        "inflate.sh": "#!/usr/bin/env bash\npython -c \"import brotli\" || pip install brotli\n",
        "inflate.py": (
            "import subprocess\nimport sys\n"
            "import brotli\nimport numpy as np\nimport torch\n"
            "subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'brotli'])\n"
        ),
        "hnerv_model.py": "import torch\n",
        "schema.py": "SCHEMA = []\nMETA = {}\n",
        "sidecar.py": "def decode_corrections(_blob):\n    return None, None\n",
    }
    manifest_files = []
    for rel, text in runtime_files.items():
        path = runtime_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        manifest_files.append(
            {
                "path": rel,
                "git_blob_sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
                "sha256": _sha256(path.read_bytes()),
                "role": "contest inflate entrypoint" if rel == "inflate.sh" else None,
            }
        )

    manifest = {
        "schema": "public_contest_reverse_engineering_intake_v1",
        "created_at_utc": "2026-05-08T09:33:23Z",
        "evidence_grade": "external_plus_empirical_custody",
        "score_claim": False,
        "entries": [
            {
                "pr_number": 102,
                "title": "hnerv_lc_v2_scale095_rplus1 submission",
                "url": "https://github.com/commaai/comma_video_compression_challenge/pull/102",
                "head_sha": "1" * 40,
                "archive": {
                    "canonical_url": "https://github.com/user-attachments/files/27369164/archive.zip",
                    "local_path": "experiments/results/pr102/archive.zip",
                    "bytes": archive_path.stat().st_size,
                    "sha256": _sha256(archive_path.read_bytes()),
                    "zip_overhead_bytes": archive_path.stat().st_size - member["compress_size"],
                    "members": [member],
                },
                "source_runtime_artifacts": {
                    "runtime_source_path": "submissions/hnerv_lc_v2_scale095_rplus1/",
                    "local_source_root": (
                        "experiments/results/pr102/source/submissions/hnerv_lc_v2_scale095_rplus1"
                    ),
                    "files": manifest_files,
                },
                "public_eval_observations": {
                    "local_exact_cuda_status": "missing",
                },
                "compliance_risks": [
                    "inflate.sh may install brotli at inflate time via pip if missing.",
                ],
            }
        ],
    }
    manifest_path = repo_root / "reverse_engineering/public_pr102_pr108_intake_20260508/manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "manifest": manifest_path,
        "archive_sha256": _sha256(archive_path.read_bytes()),
        "member_sha256": member["sha256"],
    }


def _write_archive(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED
    info.external_attr = 0o644 << 16
    with ZipFile(path, "w") as zf:
        zf.writestr(info, payload)
    with ZipFile(path) as zf:
        stored = zf.getinfo("0.bin")
        data = zf.read("0.bin")
        return {
            "name": stored.filename,
            "file_size": stored.file_size,
            "compress_size": stored.compress_size,
            "compress_type": stored.compress_type,
            "crc32": f"{stored.CRC:08x}",
            "sha256": _sha256(data),
        }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
