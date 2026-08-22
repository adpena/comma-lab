"""Behavioral controls for the DX2 CABAC coefficient receiver fold.

Every payload used here is already retained under the DX1 or DX2 custody
stores.  The tests do not create throwaway candidate bytes.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[3]
DX1 = Path("/Volumes/VertigoDataTier/pact/ddm_dx1/retained")
DX2 = Path("/Volumes/APDataStore/pact/ddm_dx2/r7")
FX5_RUNTIME = Path("/Volumes/APDataStore/pact/ddm_fx5/candidate_runtime_fx5")
DX2_RUNTIME = DX2 / "candidate_runtime_dx2"
RESULT = DX2 / "RESULT.json"

if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac import dx2_cabac_coefficients as coder  # noqa: E402

requires_custody = pytest.mark.skipif(
    not RESULT.is_file(), reason="retained DX2 r7 custody is not mounted"
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_rr5_tool():
    path = REPO / "tools/ddm_rr5_rider_apply.py"
    spec = importlib.util.spec_from_file_location("ddm_rr5_rider_apply_dx2_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def custody() -> dict:
    if not RESULT.is_file():
        pytest.skip("retained DX2 r7 custody is not mounted")
    tool = _load_rr5_tool()
    base = tool.parse_container(
        FX5_RUNTIME / "archive.zip",
        FX5_RUNTIME,
        expect_sha256=(
            "4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841"
        ),
    )
    applied = coder.apply_cabac_to_carrier_body(base.carrier_body)
    candidate = tool.parse_container(
        DX2_RUNTIME / "archive.zip",
        DX2_RUNTIME,
        expect_sha256=(
            "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"
        ),
    )
    return {
        "tool": tool,
        "base": base,
        "applied": applied,
        "candidate": candidate,
        "result": json.loads(RESULT.read_text()),
    }


@requires_custody
def test_charter_symbol_pin_is_content_hash_not_npy_container_hash() -> None:
    path = DX1 / "dx1_coded_symbols_U.int32.npy"
    file_sha = _sha(path.read_bytes())
    symbols = np.load(path, allow_pickle=False)
    content_sha = _sha(np.ascontiguousarray(symbols.astype(np.int32)).tobytes())
    assert file_sha == "8fc44020c3d5cb8ebe7d4adfabe7d1b0e05ad321f85bed03cb7086f04f201d95"
    assert content_sha == "0bfe31cf9586104f4308329fec8f76f748c56441ac5bd85b824dfcca3434db50"
    assert file_sha != content_sha


@requires_custody
def test_real_fx5_symbols_reencode_to_the_exact_measured_winner(custody) -> None:
    applied = custody["applied"]
    retained_symbols = np.load(DX1 / "dx1_coded_symbols_U.int32.npy", allow_pickle=False)
    retained_payload = (
        DX1 / "dx1_payload_adaptive-ctx_Rice_CABAC_prefix_cap8.bin"
    ).read_bytes()
    assert np.array_equal(applied["symbols"], retained_symbols)
    assert applied["cabac_payload"] == retained_payload
    assert len(retained_payload) == 9_811
    assert _sha(retained_payload) == (
        "b93131a52674abb4ada677e1b6cf08eebc6afb94381136d23d010e70a287e210"
    )


@requires_custody
def test_receiver_inverse_restores_the_fx5_carrier_byte_for_byte(custody) -> None:
    assert coder.restore_carrier_body(custody["applied"]["body"]) == (
        custody["base"].carrier_body
    )


@requires_custody
def test_corrupted_payload_negative_control_really_refuses(custody) -> None:
    corrupted = (DX2 / "retained/negative_control_corrupt_cabac.bin").read_bytes()
    with pytest.raises(coder.CabacCoefficientError):
        coder.decode_cabac_checked(corrupted, custody["applied"]["ks"])


@requires_custody
def test_candidate_changes_only_the_disjoint_carrier_section_and_signal(custody) -> None:
    base = custody["base"]
    candidate = custody["candidate"]
    assert candidate.reserved == base.reserved | coder.DX2_RESERVED_CABAC_COEFFICIENTS
    assert candidate.reserved & 0x08  # RR5 basis rider remains active.
    assert candidate.hpac_stream == base.hpac_stream
    assert candidate.semantic_stream == base.semantic_stream
    assert candidate.section_tail == base.section_tail
    assert candidate.carrier_body == custody["applied"]["body"]
    assert len(base.carrier_body) - len(candidate.carrier_body) == 18


@requires_custody
def test_archive_and_repeat_are_exactly_minus_18_bytes(custody) -> None:
    result = custody["result"]
    candidate = (DX2 / "retained/candidate_dx2_cabac.zip").read_bytes()
    repeat = (DX2 / "retained/candidate_dx2_cabac.repeat.zip").read_bytes()
    assert candidate == repeat
    assert len(candidate) == 180_368
    assert result["candidate"]["archive_delta_bytes"] == -18
    assert _sha(candidate) == (
        "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"
    )


@requires_custody
def test_real_receiver_parseback_compares_every_consumed_field(custody) -> None:
    proof = custody["tool"].receiver_decode_identity(
        FX5_RUNTIME / "archive.zip",
        DX2_RUNTIME / "archive.zip",
        FX5_RUNTIME,
        DX2_RUNTIME,
    )
    assert proof["identical"] is True
    assert proof["mismatched_fields"] == []
    assert proof["container_provenance_differs"] == ["compressed_models"]
    assert proof["container_provenance_inertness"]["all_inert"] is True


@requires_custody
def test_runtime_executes_the_reviewed_coder_bytes() -> None:
    source = (REPO / "src/tac/dx2_cabac_coefficients.py").read_bytes()
    runtime = (DX2_RUNTIME / "runtime/dx2_cabac_coefficients.py").read_bytes()
    assert runtime == source


@requires_custody
def test_runtime_tree_has_no_host_bytecode_residue() -> None:
    residue = [
        path
        for path in DX2_RUNTIME.rglob("*")
        if "__pycache__" in path.parts or path.suffix == ".pyc"
    ]
    assert residue == []


@requires_custody
def test_runtime_tree_has_no_appledouble_metadata_residue() -> None:
    assert list(DX2_RUNTIME.rglob("._*")) == []


def test_decoder_has_no_torch_or_device_branch() -> None:
    path = REPO / "src/tac/dx2_cabac_coefficients.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert "torch" not in imports
    assert "mlx" not in imports
    assert "cupy" not in imports


@requires_custody
def test_result_does_not_launder_parseback_as_the_owed_raw_gate(custody) -> None:
    gate = custody["result"]["raw_identity_gate"]
    assert gate["status"] == "NOT_RUN_HEAVY_LOCAL_SLOT_OCCUPIED"
    assert gate["required_raw_bytes"] == 3_662_409_600
    assert gate["required_contest_cuda_raw_sha256"] == (
        "6bf8acf8d4412e43f8ddf810bcf63feb6435b758196b708fd61e77fe61e79883"
    )
