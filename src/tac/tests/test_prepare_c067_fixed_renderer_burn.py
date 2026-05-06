from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[3]
PREP_PATH = REPO / "experiments" / "prepare_c067_fixed_renderer_burn.py"


def _load_prep(name: str = "_prepare_c067_fixed_renderer_burn_test") -> Any:
    spec = importlib.util.spec_from_file_location(name, PREP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_source_archive(path: Path) -> dict[str, bytes]:
    members = {
        "renderer.bin": b"QZS3renderer",
        "masks.mkv": b"mask-bytes",
        "optimized_poses.bin": b"pose-bytes",
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, payload in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            info.extra = b""
            info.comment = b""
            zf.writestr(info, payload)
    return members


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_prepare_burn_extracts_fixed_members_and_blocks_retraining_without_clearance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prep = _load_prep()
    source = tmp_path / "source.zip"
    members = _write_source_archive(source)
    monkeypatch.setattr(prep, "EXPECTED_SOURCE_ARCHIVE_SHA256", _sha256(source.read_bytes()))
    monkeypatch.setattr(
        prep,
        "EXPECTED_MEMBER_SHA256",
        {name: _sha256(payload) for name, payload in members.items()},
    )

    summary = prep.prepare_burn(
        source_archive=source,
        output_dir=tmp_path / "out",
        run_id="c067_fixed_renderer_burn_test",
        clearance_packet=tmp_path / "missing_clearance.json",
    )

    assert summary["schema"] == "c067_fixed_renderer_training_burn_prep_v1"
    assert summary["score_claim"] is False
    assert summary["remote_gpu_dispatch_performed"] is False
    assert summary["training_dispatch_gate"]["cleared_for_retraining_dispatch"] is False
    assert "lane12_l2_clearance_packet_missing_or_unreadable" in summary["training_dispatch_gate"]["blockers"]
    dryrun = summary["preflight"]["argparse_static_dryrun"]
    assert dryrun["train_renderer"]["ok"] is True
    assert dryrun["q_faithful_snapshot_loop"]["ok"] is True
    for name, payload in members.items():
        row = summary["fixed_runtime_members"][name]
        assert Path(row["path"]).read_bytes() == payload
        assert row["sha256"] == _sha256(payload)
    train = summary["commands"]["train_renderer"]
    assert "--qfaithful-training-poses" in train
    assert "--mask-noise-mkv" in train
    assert "--no-auth-eval-on-best" in train
    script = Path(summary["commands"]["shell_script"])
    assert script.exists()
    assert script.stat().st_mode & stat.S_IXUSR
    script_text = script.read_text()
    assert "'--workspace' \"$PWD\" '--python-bin'" in script_text
    assert "'$PWD'" not in script_text
    assert "SNAPSHOT_PID=$!" in script_text
    assert "cleanup_snapshot_loop()" in script_text
    assert "TRAIN_STATUS=${PIPESTATUS[0]}" in script_text
    assert "'--max-idle-polls' '720'" in script_text
    assert "'--eval-mode' 'none'" in script_text
    manifest = json.loads((tmp_path / "out" / "c067_fixed_renderer_burn_test" / "fixed_c067_renderer_burn_manifest.json").read_text())
    assert manifest == summary


def test_prepare_burn_fails_closed_on_wrong_source_sha(tmp_path: Path) -> None:
    prep = _load_prep("_prepare_c067_fixed_renderer_burn_mismatch_test")
    source = tmp_path / "source.zip"
    _write_source_archive(source)

    with pytest.raises(ValueError, match="source archive SHA mismatch"):
        prep.prepare_burn(
            source_archive=source,
            output_dir=tmp_path / "out",
            run_id="bad",
        )
