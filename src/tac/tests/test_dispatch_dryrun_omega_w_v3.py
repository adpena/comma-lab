from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

from tac.sensitivity_map import save_sensitivity_map

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "dispatch_dryrun_omega_w_v3.py"


def _load_dryrun_module():
    spec = importlib.util.spec_from_file_location(
        "dispatch_dryrun_omega_w_v3_under_test",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dryrun = _load_dryrun_module()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sensitivity(path: Path, metadata: dict[str, object]) -> None:
    save_sensitivity_map(
        path,
        {"decoder.block.weight": torch.tensor([1.0, 2.0, 3.0])},
        metadata=metadata,
    )


def _valid_metadata(source_archive: Path) -> dict[str, object]:
    archive_sha = _sha256(source_archive)
    return {
        "device": "cuda",
        "source_archive_sha256": archive_sha,
        "source_archive_bytes": source_archive.stat().st_size,
        "promotion_eligible": True,
        "official_component_response": True,
        "canonical_scorer_path": True,
        "sensitivity_source": "certified_official_component_sensitivity",
        "tag": "[certified-contest-CUDA]",
        "n_pairs": 600,
        "certification": {
            "format": "component_sensitivity_map_certification_v1",
            "component": "combined",
            "device": "cuda",
            "official_component_response": True,
            "canonical_scorer_path": True,
            "promotion_eligible": True,
            "source_map_sha256": "a" * 64,
            "official_response_curve_sha256": "b" * 64,
            "stability_sha256": "c" * 64,
            "sample_plan_sha256": "d" * 64,
            "baseline_archive_sha256": archive_sha,
            "baseline_archive_bytes": source_archive.stat().st_size,
            "contest_auth_eval_json_sha256": "e" * 64,
            "review_clean_passes": 3,
            "review_unresolved_blockers": [],
        },
    }


def test_real_sensitivity_metadata_accepts_cuda_source_archive_sha(tmp_path: Path) -> None:
    source_archive = tmp_path / "archive.zip"
    source_archive.write_bytes(b"pr106-source")
    sensitivity = tmp_path / "sensitivity_map.pt"
    _write_sensitivity(sensitivity, _valid_metadata(source_archive))

    msg = dryrun.check_real_sensitivity_metadata(sensitivity, source_archive)

    assert "real sensitivity metadata OK" in msg
    assert "certified component='combined'" in msg
    assert _sha256(source_archive)[:16] in msg


@pytest.mark.parametrize(
    ("metadata_patch", "match"),
    [
        ({"is_stub": True, "tag": "[stub-design-mode]"}, "is_stub"),
        ({"planning_only": True, "status": "planning"}, "planning"),
        ({"stale": True}, "stale"),
    ],
)
def test_real_sensitivity_metadata_rejects_stub_planning_and_stale_markers(
    tmp_path: Path,
    metadata_patch: dict[str, object],
    match: str,
) -> None:
    source_archive = tmp_path / "archive.zip"
    source_archive.write_bytes(b"pr106-source")
    sensitivity = tmp_path / "sensitivity_map.pt"
    metadata = {**_valid_metadata(source_archive), **metadata_patch}
    _write_sensitivity(sensitivity, metadata)

    with pytest.raises(dryrun.CheckFailure, match=match):
        dryrun.check_real_sensitivity_metadata(sensitivity, source_archive)


def test_real_sensitivity_metadata_rejects_source_archive_sha_mismatch(tmp_path: Path) -> None:
    source_archive = tmp_path / "archive.zip"
    source_archive.write_bytes(b"pr106-source")
    sensitivity = tmp_path / "sensitivity_map.pt"
    metadata = {
        **_valid_metadata(source_archive),
        "source_archive_sha256": "b" * 64,
    }
    _write_sensitivity(sensitivity, metadata)

    with pytest.raises(dryrun.CheckFailure, match="stale or mismatched"):
        dryrun.check_real_sensitivity_metadata(sensitivity, source_archive)


def test_real_sensitivity_metadata_requires_source_archive_sha(tmp_path: Path) -> None:
    source_archive = tmp_path / "archive.zip"
    source_archive.write_bytes(b"pr106-source")
    sensitivity = tmp_path / "sensitivity_map.pt"
    metadata = _valid_metadata(source_archive)
    del metadata["source_archive_sha256"]
    metadata["certification"] = {
        **metadata["certification"],  # type: ignore[dict-item]
        "baseline_archive_sha256": "f" * 64,
    }
    _write_sensitivity(sensitivity, metadata)

    with pytest.raises(dryrun.CheckFailure, match="source archive SHA"):
        dryrun.check_real_sensitivity_metadata(sensitivity, source_archive)


def test_real_sensitivity_metadata_requires_certification(tmp_path: Path) -> None:
    source_archive = tmp_path / "archive.zip"
    source_archive.write_bytes(b"pr106-source")
    sensitivity = tmp_path / "sensitivity_map.pt"
    metadata = _valid_metadata(source_archive)
    del metadata["certification"]
    _write_sensitivity(sensitivity, metadata)

    with pytest.raises(dryrun.CheckFailure, match="certification"):
        dryrun.check_real_sensitivity_metadata(sensitivity, source_archive)


def test_default_dryrun_preserves_stub_mode_without_real_sensitivity_gate(monkeypatch, capsys) -> None:
    called_real_gate = False

    def _stage3(
        workdir: Path,
        sensitivity_path: Path,
        source_archive: Path,
        *,
        enforce_stub_byte_exact: bool,
        allow_stub_design_mode: bool,
    ) -> str:
        (workdir / "apogee_v2_archive.zip").write_bytes(b"archive")
        return "stage3"

    monkeypatch.setattr(dryrun, "check_wrapper_exists_and_parses", lambda: "wrapper")
    monkeypatch.setattr(dryrun, "check_pr106_artifact", lambda source_archive: "archive")
    monkeypatch.setattr(dryrun, "check_sensitivity_on_disk", lambda sensitivity_path: "stub")
    monkeypatch.setattr(dryrun, "check_producer_scripts_exist", lambda: "scripts")
    monkeypatch.setattr(dryrun, "check_inflate_adapter_modules", lambda: "inflate")
    monkeypatch.setattr(dryrun, "check_stage1_extract_e2e", lambda workdir, source_archive: "stage1")
    monkeypatch.setattr(dryrun, "check_stage3_repack", _stage3)
    monkeypatch.setattr(dryrun, "check_parser_roundtrip", lambda workdir: "parser")

    def _unexpected_real_gate(*args, **kwargs):
        nonlocal called_real_gate
        called_real_gate = True
        raise AssertionError("default dry-run should not require real sensitivity")

    monkeypatch.setattr(dryrun, "check_real_sensitivity_metadata", _unexpected_real_gate)

    assert dryrun.run_dryrun() == 0
    out = capsys.readouterr().out
    assert called_real_gate is False
    assert "ready_for_remote_cuda_dispatch=false" in out
    assert "Default stub-mode is local smoke only" in out
    assert "Dispatch is GO" not in out


def test_require_real_sensitivity_adds_metadata_gate(monkeypatch, capsys) -> None:
    calls: list[str] = []

    def _stage3(
        workdir: Path,
        sensitivity_path: Path,
        source_archive: Path,
        *,
        enforce_stub_byte_exact: bool,
        allow_stub_design_mode: bool,
    ) -> str:
        (workdir / "apogee_v2_archive.zip").write_bytes(b"archive")
        return "stage3"

    monkeypatch.setattr(dryrun, "check_wrapper_exists_and_parses", lambda: "wrapper")
    monkeypatch.setattr(dryrun, "check_pr106_artifact", lambda source_archive: "archive")
    monkeypatch.setattr(dryrun, "check_sensitivity_on_disk", lambda sensitivity_path: "sensitivity")
    monkeypatch.setattr(dryrun, "check_producer_scripts_exist", lambda: "scripts")
    monkeypatch.setattr(dryrun, "check_inflate_adapter_modules", lambda: "inflate")
    monkeypatch.setattr(dryrun, "check_stage1_extract_e2e", lambda workdir, source_archive: "stage1")
    monkeypatch.setattr(dryrun, "check_stage3_repack", _stage3)
    monkeypatch.setattr(dryrun, "check_parser_roundtrip", lambda workdir: "parser")

    def _real_gate(sensitivity_path: Path, source_archive: Path) -> str:
        calls.append(f"{sensitivity_path.name}:{source_archive.name}")
        return "real"

    monkeypatch.setattr(dryrun, "check_real_sensitivity_metadata", _real_gate)

    assert dryrun.run_dryrun(require_real_sensitivity=True) == 0
    out = capsys.readouterr().out
    assert calls == [f"{dryrun.SENSITIVITY_STUB.name}:{dryrun.PR106_ARCHIVE.name}"]
    assert "ready_for_remote_cuda_dispatch=true" in out
    assert "Remote CUDA dispatch is allowed" in out
