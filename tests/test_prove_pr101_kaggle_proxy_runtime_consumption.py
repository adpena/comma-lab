from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from hashlib import sha256
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "prove_pr101_kaggle_proxy_runtime_consumption.py"
BUILDER_PATH = REPO_ROOT / "tools" / "build_pr101_kaggle_proxy_runtime_packet.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def proof_tool():
    return _load_module(TOOL_PATH, "prove_pr101_kaggle_proxy_runtime_consumption_under_test")


@pytest.fixture()
def packet_builder():
    return _load_module(BUILDER_PATH, "build_pr101_kaggle_proxy_runtime_packet_for_proof_test")


def _write_file(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.chmod(path, mode)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    _write_file(runtime / "src/model.py", "class HNeRVDecoder:\n    pass\n", 0o755)
    _write_file(runtime / "src/codec.py", "def parse_archive(data):\n    return None\n")
    return runtime


def _handoff() -> dict[str, object]:
    return {
        "schema": "pr101_kaggle_proxy_candidate_archive_builder_handoff_v1",
        "candidate_id": "proxy_cmaes_0037",
        "param_schema": "pr101_kaggle_proxy_candidate_params_v1",
        "params": {
            "bias_b": -0.79,
            "bias_g": -0.88,
            "bias_r": -1.01,
            "delta_scale": 0.009,
            "latent_delta_scale": 0.008,
            "smooth_weight": 0.019,
        },
        "evidence_boundary": {
            "score_claim": False,
            "score_claim_valid": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "exact_auth_eval_performed": False,
            "contest_cuda_auth_eval": False,
        },
    }


def _build_packet(tmp_path: Path, packet_builder) -> Path:
    runtime = _runtime_fixture(tmp_path)
    source_archive = tmp_path / "source" / "archive.zip"
    handoff_path = tmp_path / "handoff.json"
    packet_dir = tmp_path / "packet"
    _write_zip(source_archive, payload=b"original-pr101-bytes")
    _write_json(handoff_path, _handoff())
    packet_builder.build_proxy_runtime_packet(
        handoff_path=handoff_path,
        source_runtime_dir=runtime,
        source_archive=source_archive,
        packet_dir=packet_dir,
    )
    return packet_dir


def _canonical_sha(payload: object) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return sha256(text.encode("utf-8")).hexdigest()


def test_proves_only_supported_bias_params_are_runtime_consumed(
    tmp_path: Path,
    packet_builder,
    proof_tool,
) -> None:
    packet_dir = _build_packet(tmp_path, packet_builder)
    proof = proof_tool.build_runtime_consumption_proof(
        manifest_path=packet_dir / "runtime_packet_manifest.json",
    )

    proof_path = packet_dir / "runtime_consumption_proof.json"
    proof_on_disk = json.loads(proof_path.read_text(encoding="utf-8"))
    basis = dict(proof_on_disk)
    proof_sha = basis.pop("proof_sha256_excluding_self")

    assert proof == proof_on_disk
    assert proof["schema"] == "pr101_kaggle_proxy_runtime_consumption_proof_v1"
    assert proof["score_claim"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False
    assert proof["dispatch_attempted"] is False
    assert proof["runtime_consumption_proven_for_supported_bias_params"] is True
    assert proof["unsupported_proxy_params_runtime_consumed"] is False
    assert proof_sha == _canonical_sha(basis)

    assert proof["archive_unchanged_proof"]["archive_sha256"] == proof["archive_unchanged_proof"]["manifest_archive_sha256"]
    consumed = proof["inflate_static_consumption_proof"]["supported_bias_params_consumed"]
    assert {row["param"] for row in consumed} == {"bias_b", "bias_g", "bias_r"}
    assert {row["replacement"] for row in consumed} == {
        "up[:, 0, 0].add_(-1.01)",
        "up[:, 0, 2].add_(-0.79)",
        "up[:, 1, 1].add_(-0.88)",
    }
    unsupported = proof["unsupported_params_blocker_proof"]
    assert unsupported["required_blockers_present"] == [
        "unsupported_proxy_params_not_runtime_consumed",
        "delta_scale_not_runtime_consumed",
        "latent_delta_scale_not_runtime_consumed",
        "smooth_weight_not_runtime_consumed",
    ]


def test_fails_closed_when_inflate_is_tampered(
    tmp_path: Path,
    packet_builder,
    proof_tool,
) -> None:
    packet_dir = _build_packet(tmp_path, packet_builder)
    inflate_path = packet_dir / "inflate.py"
    inflate_path.write_text(
        inflate_path.read_text(encoding="utf-8").replace(
            "up[:, 0, 0].add_(-1.01)",
            "up[:, 0, 0].sub_(1.0)",
        ),
        encoding="utf-8",
    )

    with pytest.raises(proof_tool.RuntimeConsumptionProofError, match="inflate.py SHA"):
        proof_tool.build_runtime_consumption_proof(
            manifest_path=packet_dir / "runtime_packet_manifest.json",
        )
    assert not (packet_dir / "runtime_consumption_proof.json").exists()


def test_fails_closed_when_manifest_unsupported_blocker_is_tampered(
    tmp_path: Path,
    packet_builder,
    proof_tool,
) -> None:
    packet_dir = _build_packet(tmp_path, packet_builder)
    manifest_path = packet_dir / "runtime_packet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unsupported_params"]["smooth_weight"]["runtime_consumed"] = True
    basis = dict(manifest)
    basis.pop("manifest_sha256_excluding_self")
    manifest["manifest_sha256_excluding_self"] = _canonical_sha(basis)
    _write_json(manifest_path, manifest)

    with pytest.raises(proof_tool.RuntimeConsumptionProofError, match="smooth_weight.*runtime_consumed"):
        proof_tool.build_runtime_consumption_proof(manifest_path=manifest_path)
    assert not (packet_dir / "runtime_consumption_proof.json").exists()


def test_fails_closed_when_manifest_self_hash_is_stale(
    tmp_path: Path,
    packet_builder,
    proof_tool,
) -> None:
    packet_dir = _build_packet(tmp_path, packet_builder)
    manifest_path = packet_dir / "runtime_packet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["packet_archive"]["sha256"] = "0" * 64
    _write_json(manifest_path, manifest)

    with pytest.raises(proof_tool.RuntimeConsumptionProofError, match="self-hash mismatch"):
        proof_tool.build_runtime_consumption_proof(manifest_path=manifest_path)
    assert not (packet_dir / "runtime_consumption_proof.json").exists()


def test_cli_emits_false_authority_fields(tmp_path: Path, packet_builder) -> None:
    packet_dir = _build_packet(tmp_path, packet_builder)
    proof_path = packet_dir / "custom_proof.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--manifest",
            str(packet_dir / "runtime_packet_manifest.json"),
            "--proof-path",
            str(proof_path),
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
    assert len(stdout["proof_sha256_excluding_self"]) == 64
    assert proof_path.is_file()
