from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "build_pr101_kaggle_proxy_runtime_packet.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("build_pr101_kaggle_proxy_runtime_packet", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_file(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.chmod(path, mode)


def _write_zip(path: Path, member: str = "x", payload: bytes = b"archive-bytes") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr(info, payload)


def _runtime_fixture(tmp_path: Path) -> Path:
    runtime = tmp_path / "runtime"
    _write_file(runtime / "inflate.sh", "#!/usr/bin/env bash\nset -euo pipefail\n", 0o755)
    _write_file(
        runtime / "inflate.py",
        """def inflate():
    while True:
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)
        break
""",
    )
    _write_file(runtime / "src/model.py", "class HNeRVDecoder:\n    pass\n")
    _write_file(runtime / "src/codec.py", "def parse_archive(data):\n    return None\n")
    _write_file(runtime / "__pycache__/inflate.cpython-312.pyc", "cache")
    _write_file(runtime / "src/__pycache__/codec.cpython-312.pyc", "cache")
    _write_file(runtime / ".DS_Store", "finder")
    return runtime


def _handoff(params: dict[str, float] | None = None) -> dict[str, object]:
    return {
        "schema": "pr101_kaggle_proxy_candidate_archive_builder_handoff_v1",
        "candidate_id": "proxy_cmaes_0037",
        "param_schema": "pr101_kaggle_proxy_candidate_params_v1",
        "params": params
        or {
            "bias_b": -0.79,
            "bias_g": -0.88,
            "bias_r": -1.01,
            "delta_scale": 0.009,
            "latent_delta_scale": 0.008,
            "smooth_weight": 0.019,
        },
        "proxy_evidence": {
            "evidence_semantics": "kaggle_gpu_proxy_config_search_only_not_exact_auth_eval",
            "proxy_objective": 0.192,
        },
        "source_candidate": {"path": "best_proxy_candidate.json", "sha256": "0" * 64},
        "evidence_boundary": {
            "proxy_only": True,
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "exact_auth_eval_performed": False,
            "contest_cuda_auth_eval": False,
            "mps_auth_eval": False,
            "archive_zip_emitted": False,
            "inflate_runtime_emitted": False,
            "dispatch_blockers": ["real_archive_builder_handoff_only"],
        },
        "archive_builder_handoff_contract": {
            "status": "pending_real_archive_builder",
            "builder_must_consume": {
                "param_schema": "pr101_kaggle_proxy_candidate_params_v1",
                "param_keys": [
                    "bias_b",
                    "bias_g",
                    "bias_r",
                    "delta_scale",
                    "latent_delta_scale",
                    "smooth_weight",
                ],
            },
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_builds_runtime_packet_with_only_bias_params_consumed(tmp_path: Path) -> None:
    tool = _load_tool()
    runtime = _runtime_fixture(tmp_path)
    source_archive = tmp_path / "source" / "archive.zip"
    handoff_path = tmp_path / "handoff.json"
    _write_zip(source_archive, payload=b"original-pr101-bytes")
    _write_json(handoff_path, _handoff())

    manifest = tool.build_proxy_runtime_packet(
        handoff_path=handoff_path,
        source_runtime_dir=runtime,
        source_archive=source_archive,
        packet_dir=tmp_path / "packet",
    )

    packet_dir = tmp_path / "packet"
    manifest_on_disk = json.loads((packet_dir / "runtime_packet_manifest.json").read_text())
    patched_inflate = (packet_dir / "inflate.py").read_text(encoding="utf-8")

    assert manifest == manifest_on_disk
    assert (packet_dir / "archive.zip").read_bytes() == source_archive.read_bytes()
    assert manifest["schema"] == "pr101_kaggle_proxy_runtime_packet_v1"
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["scorers_invoked"] is False
    assert manifest["archive_changed"] is False
    assert manifest["archive_unchanged_sha256"] == manifest["source_archive"]["sha256"]
    assert manifest["archive_unchanged_sha256"] == manifest["packet_archive"]["sha256"]

    assert "up[:, 0, 0].add_(-1.01)" in patched_inflate
    assert "up[:, 0, 2].add_(-0.79)" in patched_inflate
    assert "up[:, 1, 1].add_(-0.88)" in patched_inflate
    assert "sub_(1.0)" not in patched_inflate

    consumed = manifest["runtime_consumed_params"]
    assert consumed == {"bias_b": -0.79, "bias_g": -0.88, "bias_r": -1.01}
    patch_rows = manifest["runtime_patch"]["runtime_consumed_params"]
    assert {row["param"]: row["slot"] for row in patch_rows} == {
        "bias_r": "up[:, 0, 0]",
        "bias_b": "up[:, 0, 2]",
        "bias_g": "up[:, 1, 1]",
    }

    assert set(manifest["unsupported_params"]) == {
        "delta_scale",
        "latent_delta_scale",
        "smooth_weight",
    }
    assert all(row["runtime_consumed"] is False for row in manifest["unsupported_params"].values())
    assert "unsupported_proxy_params_not_runtime_consumed" in manifest["blockers"]
    assert "delta_scale_not_runtime_consumed" in manifest["blockers"]
    assert "latent_delta_scale_not_runtime_consumed" in manifest["blockers"]
    assert "smooth_weight_not_runtime_consumed" in manifest["blockers"]
    assert not (packet_dir / "__pycache__").exists()
    assert not (packet_dir / "src/__pycache__").exists()
    assert not (packet_dir / ".DS_Store").exists()


def test_refuses_handoff_that_claims_exact_eval_authority(tmp_path: Path) -> None:
    tool = _load_tool()
    runtime = _runtime_fixture(tmp_path)
    source_archive = tmp_path / "source/archive.zip"
    _write_zip(source_archive)
    handoff = _handoff()
    handoff["evidence_boundary"]["ready_for_exact_eval_dispatch"] = True
    handoff_path = tmp_path / "handoff.json"
    _write_json(handoff_path, handoff)

    with pytest.raises(tool.ProxyRuntimePacketError, match="ready_for_exact_eval_dispatch"):
        tool.build_proxy_runtime_packet(
            handoff_path=handoff_path,
            source_runtime_dir=runtime,
            source_archive=source_archive,
            packet_dir=tmp_path / "packet",
        )


def test_refuses_missing_or_nonfinite_required_params(tmp_path: Path) -> None:
    tool = _load_tool()
    runtime = _runtime_fixture(tmp_path)
    source_archive = tmp_path / "source/archive.zip"
    _write_zip(source_archive)

    missing = _handoff()
    del missing["params"]["smooth_weight"]
    missing_path = tmp_path / "missing.json"
    _write_json(missing_path, missing)
    with pytest.raises(tool.ProxyRuntimePacketError, match="missing required keys"):
        tool.build_proxy_runtime_packet(
            handoff_path=missing_path,
            source_runtime_dir=runtime,
            source_archive=source_archive,
            packet_dir=tmp_path / "packet-missing",
        )

    bad_params = dict(_handoff()["params"])
    bad_params["bias_r"] = math.inf
    nonfinite_path = tmp_path / "nonfinite.json"
    _write_json(nonfinite_path, _handoff(bad_params))
    with pytest.raises(tool.ProxyRuntimePacketError, match="params.bias_r must be finite"):
        tool.build_proxy_runtime_packet(
            handoff_path=nonfinite_path,
            source_runtime_dir=runtime,
            source_archive=source_archive,
            packet_dir=tmp_path / "packet-nonfinite",
        )


def test_refuses_runtime_without_exact_pr101_bias_lines(tmp_path: Path) -> None:
    tool = _load_tool()
    runtime = _runtime_fixture(tmp_path)
    (runtime / "inflate.py").write_text("def inflate():\n    return None\n", encoding="utf-8")
    source_archive = tmp_path / "source/archive.zip"
    handoff_path = tmp_path / "handoff.json"
    _write_zip(source_archive)
    _write_json(handoff_path, _handoff())

    with pytest.raises(tool.ProxyRuntimePacketError, match="expected exactly one PR101 bias line"):
        tool.build_proxy_runtime_packet(
            handoff_path=handoff_path,
            source_runtime_dir=runtime,
            source_archive=source_archive,
            packet_dir=tmp_path / "packet",
        )


def test_cli_outputs_manifest_without_dispatch_or_score_claim(tmp_path: Path) -> None:
    runtime = _runtime_fixture(tmp_path)
    source_archive = tmp_path / "source/archive.zip"
    handoff_path = tmp_path / "handoff.json"
    packet_dir = tmp_path / "packet"
    _write_zip(source_archive)
    _write_json(handoff_path, _handoff())

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--handoff",
            str(handoff_path),
            "--source-runtime-dir",
            str(runtime),
            "--source-archive",
            str(source_archive),
            "--packet-dir",
            str(packet_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = json.loads(proc.stdout)
    assert stdout["candidate_id"] == "proxy_cmaes_0037"
    assert stdout["score_claim"] is False
    assert stdout["ready_for_exact_eval_dispatch"] is False
    assert stdout["dispatch_attempted"] is False
    assert len(stdout["runtime_tree_sha256"]) == 64
    assert len(stdout["archive_unchanged_sha256"]) == 64
    assert (packet_dir / "runtime_packet_manifest.json").is_file()
