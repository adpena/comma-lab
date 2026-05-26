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
import torch.nn.functional as F

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
    _write_file(
        runtime / "inflate.sh",
        """#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"
mkdir -p "$OUTPUT_DIR"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  python "$HERE/inflate.py" "$DATA_DIR/x" "$OUTPUT_DIR/${BASE}.raw"
done < "$FILE_LIST"
""",
        0o755,
    )
    _write_file(
        runtime / "inflate.py",
        """import sys
from pathlib import Path

import torch
import torch.nn.functional as F

def parse_archive(archive_bytes):
    raise RuntimeError("test fixture expects proof tool to stub parse_archive")

class HNeRVDecoder:
    def __init__(self, **kwargs):
        raise RuntimeError("test fixture expects proof tool to stub HNeRVDecoder")

CAMERA_H = 874
CAMERA_W = 1164

def inflate(src_bin: str, dst_raw: str):
    with open(src_bin, "rb") as f:
        archive_bytes = f.read()
    decoder_sd, latents, meta = parse_archive(archive_bytes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()
    latents = latents.to(device)
    n_pairs = meta["n_pairs"]
    eval_h, eval_w = meta["eval_size"]
    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            batch = j - i
            decoded = decoder(latents[i:j])
            flat = decoded.reshape(batch * 2, 3, eval_h, eval_w)
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)
            frames = (
                up.reshape(batch * 2, 3, CAMERA_H, CAMERA_W)
                .clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            fout.write(frames.tobytes())
            n += batch * 2
    return n

if __name__ == "__main__":
    inflate(sys.argv[1], sys.argv[2])
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


def _mode_string(path: Path) -> str:
    return f"{path.stat().st_mode & 0o777:04o}"


def _refresh_manifest_runtime_file(manifest_path: Path, relpath: str) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet_dir = Path(manifest["packet_dir"])
    target = packet_dir / relpath
    for row in manifest["runtime_custody"]["runtime_files"]:
        if row["relpath"] == relpath:
            row["bytes"] = target.stat().st_size
            row["mode"] = _mode_string(target)
            row["sha256"] = sha256(target.read_bytes()).hexdigest()
            break
    else:  # pragma: no cover - fixture sanity guard
        raise AssertionError(f"runtime file not found in manifest: {relpath}")
    basis = [
        {
            "bytes": row["bytes"],
            "mode": row["mode"],
            "relpath": row["relpath"],
            "sha256": row["sha256"],
        }
        for row in sorted(manifest["runtime_custody"]["runtime_files"], key=lambda item: item["relpath"])
    ]
    manifest["runtime_custody"]["runtime_tree_sha256"] = _canonical_sha(basis)
    manifest_basis = dict(manifest)
    manifest_basis.pop("manifest_sha256_excluding_self")
    manifest["manifest_sha256_excluding_self"] = _canonical_sha(manifest_basis)
    _write_json(manifest_path, manifest)


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
    assert proof["proof_kind"] == "static_bias_patch_plus_local_wrapper_route_plus_no_scorer_bias_runtime_v1"
    assert proof["score_claim"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False
    assert proof["dispatch_attempted"] is False
    assert proof["supported_bias_params_static_patch_proven"] is True
    assert proof["inflate_sh_routes_to_packet_inflate_py"] is True
    assert proof["runtime_consumption_proven_for_supported_bias_params"] is True
    assert proof["scorers_invoked"] is False
    assert proof["gpu_required"] is False
    assert proof_sha == _canonical_sha(basis)

    assert proof["archive_unchanged_proof"]["archive_sha256"] == proof["archive_unchanged_proof"]["manifest_archive_sha256"]
    consumed = proof["inflate_static_bias_patch_proof"]["supported_bias_params_consumed"]
    assert {row["param"] for row in consumed} == {"bias_b", "bias_g", "bias_r"}
    assert {row["replacement"] for row in consumed} == {
        "up[:, 0, 0].add_(-1.01)",
        "up[:, 0, 2].add_(-0.79)",
        "up[:, 1, 1].add_(-0.88)",
    }
    candidate_contract = proof["candidate_contract_proof"]
    assert candidate_contract["candidate_param_schema"] == "pr101_kaggle_proxy_bias_runtime_params_v1"
    assert set(candidate_contract["candidate_params"]) == {"bias_b", "bias_g", "bias_r"}
    assert candidate_contract["candidate_params"] == candidate_contract["runtime_consumed_params"]
    assert set(candidate_contract["ignored_legacy_handoff_params"]) == {
        "delta_scale",
        "latent_delta_scale",
        "smooth_weight",
    }
    assert candidate_contract["removed_unsupported_param_blockers_absent"] == [
        "unsupported_proxy_params_not_runtime_consumed",
        "delta_scale_not_runtime_consumed",
        "latent_delta_scale_not_runtime_consumed",
        "smooth_weight_not_runtime_consumed",
    ]
    assert "unsupported_params_blocker_proof" not in proof
    assert "unsupported_proxy_params_not_runtime_consumed" not in proof["dispatch_blockers"]
    route = proof["inflate_wrapper_route_proof"]
    assert route["proof_kind"] == "local_no_scorer_wrapper_route_probe_v1"
    assert route["wrapper_invoked_packet_inflate_py"] is True
    assert route["packet_inflate_py_sha256"] == proof["inflate_static_bias_patch_proof"]["inflate_sha256"]
    assert route["observed_argv_shape"] == ["inflate.py", "src_bin", "dst_raw"]
    assert route["observed_src_basename"] == "x"
    assert route["observed_dst_basename"] == "0.raw"
    assert route["scorers_invoked"] is False
    assert route["gpu_required"] is False
    runtime = proof["inflate_runtime_bias_logic_proof"]
    assert runtime["proof_kind"] == "local_no_scorer_real_inflate_bias_runtime_probe_v1"
    assert runtime["packet_inflate_function_executed"] is True
    assert runtime["supported_bias_params_consumed_by_runtime_logic"] is True
    assert runtime["probe_camera_shape"] == [2, 2]
    assert runtime["probe_output_bytes"] == 24
    assert runtime["probe_n_frames"] == 2
    assert runtime["parse_archive_stubbed"] is True
    assert runtime["decoder_stubbed"] is True
    assert runtime["scorers_invoked"] is False
    assert runtime["gpu_required"] is False
    assert runtime["blocked_scorer_import_attempts"] == []
    slot_proofs = {row["param"]: row for row in runtime["supported_bias_slot_proofs"]}
    assert set(slot_proofs) == {"bias_b", "bias_g", "bias_r"}
    assert slot_proofs["bias_r"]["expected_delta"] == -1.01
    assert slot_proofs["bias_b"]["expected_delta"] == -0.79
    assert slot_proofs["bias_g"]["expected_delta"] == -0.88
    assert all(row["max_abs_error"] == 0.0 for row in slot_proofs.values())


def test_runtime_bias_probe_restores_global_interpolate(
    tmp_path: Path,
    packet_builder,
    proof_tool,
) -> None:
    original_interpolate = F.interpolate
    packet_dir = _build_packet(tmp_path, packet_builder)

    proof_tool.build_runtime_consumption_proof(
        manifest_path=packet_dir / "runtime_packet_manifest.json",
    )

    assert F.interpolate is original_interpolate


def test_fails_closed_when_noop_inflate_sh_does_not_route_to_packet_inflate_py(
    tmp_path: Path,
    packet_builder,
    proof_tool,
) -> None:
    packet_dir = _build_packet(tmp_path, packet_builder)
    inflate_sh = packet_dir / "inflate.sh"
    inflate_sh.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n", encoding="utf-8")
    os.chmod(inflate_sh, 0o755)
    _refresh_manifest_runtime_file(packet_dir / "runtime_packet_manifest.json", "inflate.sh")

    with pytest.raises(
        proof_tool.RuntimeConsumptionProofError,
        match=r"did not invoke packet inflate\.py",
    ):
        proof_tool.build_runtime_consumption_proof(
            manifest_path=packet_dir / "runtime_packet_manifest.json",
        )
    assert not (packet_dir / "runtime_consumption_proof.json").exists()


def test_minimal_wrapper_invoking_python_inflate_py_passes_wrapper_route_proof(
    tmp_path: Path,
    packet_builder,
    proof_tool,
) -> None:
    packet_dir = _build_packet(tmp_path, packet_builder)

    proof = proof_tool.build_runtime_consumption_proof(
        manifest_path=packet_dir / "runtime_packet_manifest.json",
    )

    assert proof["inflate_wrapper_route_proof"]["wrapper_invoked_packet_inflate_py"] is True
    assert proof["inflate_sh_routes_to_packet_inflate_py"] is True
    assert proof["runtime_consumption_proven_for_supported_bias_params"] is True


def test_fails_closed_when_bias_lines_are_static_but_not_executed_by_inflate(
    tmp_path: Path,
    packet_builder,
    proof_tool,
) -> None:
    runtime = _runtime_fixture(tmp_path)
    _write_file(
        runtime / "inflate.py",
        """def unused_bias_patch_surface(up):
    if True:
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)

def inflate(src_bin: str, dst_raw: str):
    open(dst_raw, "wb").write(b"noop")
    return 0

if __name__ == "__main__":
    import sys
    inflate(sys.argv[1], sys.argv[2])
""",
    )
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

    with pytest.raises(
        proof_tool.RuntimeConsumptionProofError,
        match=r"did not exercise F\.interpolate",
    ):
        proof_tool.build_runtime_consumption_proof(
            manifest_path=packet_dir / "runtime_packet_manifest.json",
        )
    assert not (packet_dir / "runtime_consumption_proof.json").exists()


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

    with pytest.raises(proof_tool.RuntimeConsumptionProofError, match=r"inflate\.py SHA"):
        proof_tool.build_runtime_consumption_proof(
            manifest_path=packet_dir / "runtime_packet_manifest.json",
        )
    assert not (packet_dir / "runtime_consumption_proof.json").exists()


def test_fails_closed_when_manifest_advertises_removed_unsupported_params(
    tmp_path: Path,
    packet_builder,
    proof_tool,
) -> None:
    packet_dir = _build_packet(tmp_path, packet_builder)
    manifest_path = packet_dir / "runtime_packet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unsupported_params"] = {
        "smooth_weight": {
            "value": 0.019,
            "runtime_consumed": False,
            "blocker": "smooth_weight_not_runtime_consumed",
        }
    }
    basis = dict(manifest)
    basis.pop("manifest_sha256_excluding_self")
    manifest["manifest_sha256_excluding_self"] = _canonical_sha(basis)
    _write_json(manifest_path, manifest)

    with pytest.raises(proof_tool.RuntimeConsumptionProofError, match="must not advertise unsupported_params"):
        proof_tool.build_runtime_consumption_proof(manifest_path=manifest_path)
    assert not (packet_dir / "runtime_consumption_proof.json").exists()


def test_fails_closed_when_manifest_reintroduces_unsupported_param_blocker(
    tmp_path: Path,
    packet_builder,
    proof_tool,
) -> None:
    packet_dir = _build_packet(tmp_path, packet_builder)
    manifest_path = packet_dir / "runtime_packet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["blockers"].append("unsupported_proxy_params_not_runtime_consumed")
    basis = dict(manifest)
    basis.pop("manifest_sha256_excluding_self")
    manifest["manifest_sha256_excluding_self"] = _canonical_sha(basis)
    _write_json(manifest_path, manifest)

    with pytest.raises(proof_tool.RuntimeConsumptionProofError, match="removed unsupported proxy params"):
        proof_tool.build_runtime_consumption_proof(manifest_path=manifest_path)
    assert not (packet_dir / "runtime_consumption_proof.json").exists()


def test_fails_closed_when_unconsumed_legacy_param_is_advertised_as_candidate(
    tmp_path: Path,
    packet_builder,
    proof_tool,
) -> None:
    packet_dir = _build_packet(tmp_path, packet_builder)
    manifest_path = packet_dir / "runtime_packet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["candidate_params"]["smooth_weight"] = 0.019
    basis = dict(manifest)
    basis.pop("manifest_sha256_excluding_self")
    manifest["manifest_sha256_excluding_self"] = _canonical_sha(basis)
    _write_json(manifest_path, manifest)

    with pytest.raises(proof_tool.RuntimeConsumptionProofError, match="candidate_params must contain only"):
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
    assert stdout["proof_kind"] == "static_bias_patch_plus_local_wrapper_route_plus_no_scorer_bias_runtime_v1"
    assert stdout["score_claim"] is False
    assert stdout["ready_for_exact_eval_dispatch"] is False
    assert stdout["dispatch_attempted"] is False
    assert stdout["inflate_sh_routes_to_packet_inflate_py"] is True
    assert stdout["runtime_consumption_proven_for_supported_bias_params"] is True
    assert "unsupported_proxy_params_not_runtime_consumed" not in stdout["dispatch_blockers"]
    assert "full_inflate_runtime_not_executed_by_this_probe" not in stdout["dispatch_blockers"]
    assert "no_scorer_runtime_probe_not_contest_auth_eval" in stdout["dispatch_blockers"]
    assert len(stdout["proof_sha256_excluding_self"]) == 64
    assert proof_path.is_file()
