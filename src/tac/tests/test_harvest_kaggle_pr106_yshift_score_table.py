# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from tac.repo_io import read_json, write_json
from tac.tests.tool_loader import load_repo_tool

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    return load_repo_tool(
        REPO_ROOT,
        "tools/harvest_kaggle_pr106_yshift_score_table.py",
        "harvest_kaggle_pr106_yshift_score_table_test",
    )


def test_harvester_writes_manifest_and_uses_yshift_prefix(monkeypatch, tmp_path: Path, capsys) -> None:
    module = _load_tool()
    calls: dict[str, object] = {}

    def fake_download(**kwargs):
        calls["download"] = kwargs
        download_dir = Path(kwargs["download_dir"])
        (download_dir / "pr106_yshift_score_table/yshift_run/score_table/score_table_manifest.json").parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        write_json(
            download_dir / "pr106_yshift_score_table/yshift_run/score_table/score_table_manifest.json",
            {"score_claim": False},
        )
        return {"downloaded": []}

    def fake_ingest(**kwargs):
        calls["ingest"] = kwargs
        return {"evidence_dir": str(Path(kwargs["output_root"]) / "run-test")}

    monkeypatch.setattr(module, "download_kernel_outputs", fake_download)
    monkeypatch.setattr(module, "ingest_downloaded_outputs", fake_ingest)

    rc = module.main(
        [
            "--run-id",
            "run-test",
            "--download-dir",
            str(tmp_path / "download"),
            "--output-root",
            str(tmp_path / "ingested"),
        ]
    )

    assert rc == 0
    manifest_path = tmp_path / "download/kaggle_pr106_yshift_score_table_manifest.json"
    manifest = read_json(manifest_path)
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["kernel_ref"] == module.DEFAULT_KERNEL_REF
    assert calls["download"]["include_patterns"] == [
        r"^pr106_yshift_score_table/",
        r"^pact_pr106_yshift_workspace/inputs/pr106_archive\.zip$",
    ]
    assert calls["ingest"]["manifest_path"] == manifest_path
    assert "run-test" in capsys.readouterr().out


def test_harvester_can_ingest_existing_download_without_api(monkeypatch, tmp_path: Path) -> None:
    module = _load_tool()
    calls: dict[str, object] = {}

    def forbidden_download(**_kwargs):
        raise AssertionError("download should not run under --no-download")

    def fake_ingest(**kwargs):
        calls["ingest"] = kwargs
        return {"evidence_dir": str(Path(kwargs["output_root"]) / "run-existing")}

    monkeypatch.setattr(module, "download_kernel_outputs", forbidden_download)
    monkeypatch.setattr(module, "ingest_downloaded_outputs", fake_ingest)

    rc = module.main(
        [
            "--run-id",
            "run-existing",
            "--download-dir",
            str(tmp_path / "download"),
            "--output-root",
            str(tmp_path / "ingested"),
            "--no-download",
        ]
    )

    assert rc == 0
    assert calls["ingest"]["download_dir"] == tmp_path / "download"
