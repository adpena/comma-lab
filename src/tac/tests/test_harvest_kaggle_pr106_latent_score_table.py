from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from tac.repo_io import read_json, write_json

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "harvest_kaggle_pr106_latent_score_table.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "harvest_kaggle_pr106_latent_score_table_test",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_harvester_writes_manifest_and_uses_latent_prefix(monkeypatch, tmp_path: Path, capsys) -> None:
    module = _load_tool()
    calls: dict[str, object] = {}

    def fake_download(**kwargs):
        calls["download"] = kwargs
        download_dir = Path(kwargs["download_dir"])
        (download_dir / "pr106_latent_score_table/latent_run/score_table/score_table_manifest.json").parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        write_json(
            download_dir / "pr106_latent_score_table/latent_run/score_table/score_table_manifest.json",
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
    manifest_path = tmp_path / "download/kaggle_pr106_latent_score_table_manifest.json"
    manifest = read_json(manifest_path)
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["kernel_ref"] == module.DEFAULT_KERNEL_REF
    assert calls["download"]["include_patterns"] == [r"^pr106_latent_score_table/"]
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
