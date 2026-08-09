#!/usr/bin/env python3
"""Verify the scorer-free invariants of the explicit PR130 CPU runtime copy."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

import torch

ARCHIVE_SHA256 = "0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd"
ARCHIVE_BYTES = 191_052
MEMBER_SHA256 = "fcc6a3c242106350077d3c328a2c07c8994c86d19b1a361f8507917de6ba3d84"
MEMBER_BYTES = 190_952
TOKEN_STREAM_SHA256 = "948379872ff81a4e5d948ec301c143be00ebd0033544c8abdfb4af0f4c4a15eb"
SEMANTIC_STATE_SHA256 = "de0e5fc616b75eb0bdb55528aa61a7405eebbd3a81064f513adfbb69b33105ba"
BASIS_TENSOR_SHA256 = "7a6576e991a068e084ffc12f6377b9bfcc00fd2529eb8df27c424921f3c3933b"
COEFFICIENT_TENSOR_SHA256 = "dee2587ec99eea45e76ebd68eaaad3e3ae52d51e0a9b673103be8d4128e07ca8"
N600_TOKEN_SHA256 = "c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return sha256_bytes(value.detach().cpu().contiguous().numpy().tobytes())


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def expected_cpu_copy(reference: str) -> str:
    updated = reference.replace("import math\nimport sys\n", "import math\nimport os\nimport sys\n", 1)
    marker = "\ndef main():\n"
    device_function = '''
def resolve_device() -> torch.device:
    """Resolve the explicitly requested PR130 inflate rail.

    Auto-selection preserves PR130's CUDA behavior on the T4 rail and selects
    CPU on the CPU-only runner.  Either project-specific or canonical harness
    policy can make that choice explicit.
    """

    requested = os.environ.get(
        "PR130_INFLATE_DEVICE",
        os.environ.get("PACT_INFLATE_DEVICE", "auto"),
    ).strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise RuntimeError(
            "PR130_INFLATE_DEVICE/PACT_INFLATE_DEVICE must be exactly "
            "'auto', 'cpu', or 'cuda'; "
            f"got {requested!r}"
        )
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "PR130 CUDA inflate was requested but CUDA is unavailable"
        )
    return torch.device(requested)


def main():
'''
    if marker not in updated:
        raise AssertionError("reference inflate.py lacks the expected main marker")
    updated = updated.replace(marker, device_function, 1)
    old_device = '''    if not torch.cuda.is_available():
        raise RuntimeError("semantic_pose_landslide requires the official GPU rail")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
'''
    new_device = '''    device = resolve_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    print(f"PR130_INFLATE_DEVICE_RESOLVED={device.type}", flush=True)
'''
    if old_device not in updated:
        raise AssertionError("reference inflate.py lacks the expected CUDA-only block")
    updated = updated.replace(old_device, new_device, 1)
    old_cache = "    torch.cuda.empty_cache()\n"
    new_cache = "    if device.type == \"cuda\":\n        torch.cuda.empty_cache()\n"
    if old_cache not in updated:
        raise AssertionError("reference inflate.py lacks the expected CUDA cache call")
    return updated.replace(old_cache, new_cache, 1)


def expected_entrypoint_copy(reference: str) -> str:
    old = '''cd -- "$SCRIPT_DIR"
exec "$PYBIN" inflate.py "$@"
'''
    new = '''if [ "$#" -ne 3 ]; then
    echo "usage: inflate.sh <archive-dir> <output-dir> <video-names-file>" >&2
    exit 64
fi

ARCHIVE_DIR=$1
OUTPUT_DIR=$2
VIDEO_NAMES_FILE=$3
mkdir -p -- "$OUTPUT_DIR"

cd -- "$SCRIPT_DIR"
while IFS= read -r video_name || [ -n "$video_name" ]; do
    [ -n "$video_name" ] || continue
    base=${video_name%.*}
    "$PYBIN" inflate.py "$ARCHIVE_DIR" "$base" "$OUTPUT_DIR/$base.raw"
done < "$VIDEO_NAMES_FILE"
'''
    if old not in reference:
        raise AssertionError("reference inflate.sh lacks the pass-through entrypoint")
    return reference.replace(old, new, 1)


def expected_dependency_manifest_copy(reference: str) -> dict[str, object]:
    expected = json.loads(reference)
    expected["borrowed_substrate_accounting"]["ddm_dv1_modified_files"] = {
        "inflate.py": (
            "adds explicit-or-auto CPU/CUDA rail selection while retaining "
            "TF32-off on CUDA"
        ),
        "inflate.sh": (
            "restores the contest three-argument loop from the read-only "
            "PR130 intake after dependency closure"
        ),
    }
    return expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--reference-runtime", type=Path, required=True)
    parser.add_argument("--cpr1-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive = args.archive.resolve()
    runtime = args.runtime.resolve()
    reference_runtime = args.reference_runtime.resolve()
    verification_path = args.cpr1_verification.resolve()
    output = args.output.resolve()

    assert archive.stat().st_size == ARCHIVE_BYTES
    assert sha256_file(archive) == ARCHIVE_SHA256
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == ["p"]
        payload = bundle.read("p")
    assert len(payload) == MEMBER_BYTES
    assert sha256_bytes(payload) == MEMBER_SHA256

    copied_files: dict[str, dict[str, object]] = {}
    for copied in sorted(runtime.iterdir()):
        if not copied.is_file():
            continue
        reference = reference_runtime / copied.name
        if not reference.is_file():
            raise AssertionError(f"CPU runtime has an untracked copied member: {copied.name}")
        copied_hash = sha256_file(copied)
        reference_hash = sha256_file(reference)
        identical = copied_hash == reference_hash
        if copied.name == "inflate.py":
            expected = expected_cpu_copy(reference.read_text())
            if copied.read_text() != expected:
                raise AssertionError("CPU inflate.py differs beyond the explicit device patch")
            change = "explicit_cpu_device_selector_only"
        elif copied.name == "inflate.sh":
            expected = expected_entrypoint_copy(reference.read_text())
            if copied.read_text() != expected:
                raise AssertionError("CPU inflate.sh differs beyond the contest-loop repair")
            change = "contest_three_argument_loop_restored"
        elif copied.name == "runtime-dependencies.json":
            expected = expected_dependency_manifest_copy(reference.read_text())
            if json.loads(copied.read_text()) != expected:
                raise AssertionError("runtime dependency manifest has undeclared drift")
            change = "ddm_dv1_runtime_changes_declared"
        else:
            if not identical:
                raise AssertionError(f"non-inflate runtime member drifted: {copied.name}")
            change = "byte_identical_copy"
        copied_files[copied.name] = {
            "copy_sha256": copied_hash,
            "reference_sha256": reference_hash,
            "byte_identical": identical,
            "disposition": change,
        }
    shell_syntax = subprocess.run(
        ["sh", "-n", str(runtime / "inflate.sh")],
        check=False,
        capture_output=True,
        text=True,
    )
    if shell_syntax.returncode != 0:
        raise AssertionError(f"CPU inflate.sh syntax failed: {shell_syntax.stderr}")

    sys.path.insert(0, str(runtime))
    inflate = importlib.import_module("inflate")
    receiver = importlib.import_module("receiver")

    previous_pr130 = os.environ.pop("PR130_INFLATE_DEVICE", None)
    previous_pact = os.environ.pop("PACT_INFLATE_DEVICE", None)
    try:
        auto_device = inflate.resolve_device()
        os.environ["PR130_INFLATE_DEVICE"] = "cpu"
        explicit_cpu_device = inflate.resolve_device()
        os.environ.pop("PR130_INFLATE_DEVICE")
        os.environ["PACT_INFLATE_DEVICE"] = "cpu"
        canonical_cpu_device = inflate.resolve_device()
        os.environ["PR130_INFLATE_DEVICE"] = "invalid"
        invalid_refused = False
        try:
            inflate.resolve_device()
        except RuntimeError as error:
            invalid_refused = "must be exactly" in str(error)
    finally:
        if previous_pact is None:
            os.environ.pop("PACT_INFLATE_DEVICE", None)
        else:
            os.environ["PACT_INFLATE_DEVICE"] = previous_pact
        if previous_pr130 is None:
            os.environ.pop("PR130_INFLATE_DEVICE", None)
        else:
            os.environ["PR130_INFLATE_DEVICE"] = previous_pr130
    assert explicit_cpu_device.type == "cpu"
    assert canonical_cpu_device.type == "cpu"
    assert invalid_refused
    assert auto_device.type == ("cuda" if torch.cuda.is_available() else "cpu")

    parts = receiver.split_payload(payload)
    assert sha256_bytes(parts.tokens) == TOKEN_STREAM_SHA256
    decoded_models = receiver.decode_models(parts.models, model_codec=parts.model_codec)
    models_raw = decoded_models.raw
    semantic_bytes = int.from_bytes(models_raw[:4], byteorder="little")
    carrier_bytes = int.from_bytes(models_raw[4:8], byteorder="little")
    semantic_pose_bytes = 8 + semantic_bytes + carrier_bytes
    semantic, basis, coefficient = inflate.unpack_semantic_pose(
        models_raw[:semantic_pose_bytes]
    )
    semantic_hash = state_sha256(semantic.state_dict())
    basis_hash = tensor_sha256(basis)
    coefficient_hash = tensor_sha256(coefficient)
    assert semantic_hash == SEMANTIC_STATE_SHA256
    assert basis_hash == BASIS_TENSOR_SHA256
    assert coefficient_hash == COEFFICIENT_TENSOR_SHA256
    hpac = inflate.load_hpac(models_raw[semantic_pose_bytes:], torch.device("cpu"))
    hpac_state_hash = state_sha256(hpac.state_dict())

    historical = json.loads(verification_path.read_text())
    cpu_decode = historical["exact_equivalence"]["cpu_full_token_decode"]
    inherited_n600_ok = (
        cpu_decode["status"] == "passed"
        and cpu_decode["device"] == "cpu"
        and cpu_decode["frames"] == 600
        and cpu_decode["tensor_shape"] == [600, 384, 512]
        and cpu_decode["decoded_raw_token_sha256"] == N600_TOKEN_SHA256
        and historical["exact_equivalence"]["decoded_raw_token_sha256_expected"]
        == N600_TOKEN_SHA256
    )
    assert inherited_n600_ok

    result = {
        "schema": "ddm_dv1_cpu_runtime_verification.v1",
        "measured_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "producer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "argv": list(sys.argv),
        },
        "score_claim": False,
        "scorer_invoked": False,
        "eval_invoked": False,
        "dispatch_invoked": False,
        "axis": "[macOS-CPU scorer-free build verification]",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "archive": {
            "path": str(archive),
            "bytes": ARCHIVE_BYTES,
            "sha256": ARCHIVE_SHA256,
            "member_bytes": MEMBER_BYTES,
            "member_sha256": MEMBER_SHA256,
            "token_stream_sha256": TOKEN_STREAM_SHA256,
            "model_stream_sha256": sha256_bytes(parts.models),
            "decoded_models_raw_sha256": sha256_bytes(models_raw),
            "model_codec": parts.model_codec,
            "token_codec": parts.token_codec,
        },
        "copy_surface": {
            "cpu_runtime": str(runtime),
            "reference_runtime": str(reference_runtime),
            "files": copied_files,
            "only_changes": [
                "inflate.py explicit-or-auto CPU/CUDA device selector",
                "inflate.sh contest three-argument per-video loop restoration",
                "runtime-dependencies.json declaration of those two changes",
            ],
            "inflate_sh_syntax_check": "PASS",
        },
        "device_selector": {
            "auto_resolves_to_available_official_rail": auto_device.type,
            "explicit_cpu_resolves": explicit_cpu_device.type == "cpu",
            "canonical_harness_cpu_resolves": canonical_cpu_device.type == "cpu",
            "invalid_value_refused": invalid_refused,
        },
        "fresh_scorer_free_model_checks": {
            "semantic_state_sha256": semantic_hash,
            "basis_tensor_sha256": basis_hash,
            "coefficient_tensor_sha256": coefficient_hash,
            "hpac_state_sha256": hpac_state_hash,
            "semantic_matches_published_exact_equivalence": True,
            "basis_matches_published_exact_equivalence": True,
            "coefficient_matches_published_exact_equivalence": True,
        },
        "inherited_n600_cpu_token_decode": {
            "classification": "VERIFIED_RETAINED_PRIMARY_ARTIFACT_NOT_FRESH_ON_CPU_COPY",
            "source_path": str(verification_path),
            "source_sha256": sha256_file(verification_path),
            "status": cpu_decode["status"],
            "frames": cpu_decode["frames"],
            "tensor_shape": cpu_decode["tensor_shape"],
            "wall_seconds": cpu_decode["wall_seconds"],
            "decoded_raw_token_sha256": cpu_decode["decoded_raw_token_sha256"],
            "matches_expected_token_sha256": inherited_n600_ok,
        },
        "unproved_boundaries": [
            "No fresh full-n600 token decode was run on this CPU copy because ddm_dt1 owns the live decoder measurement.",
            "No CPU raw-video render was run; CPU-vs-CUDA frame equality is unmeasured.",
            "No CPU or CUDA scorer was run; scorer-numerics equality is unmeasured and not implied by TF32-off.",
        ],
        "verdict": "PASS_BUILD_AND_SCORER_FREE_PARSE_MODEL_EQUIVALENCE",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "verdict": result["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
