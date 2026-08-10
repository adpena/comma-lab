"""Closure tests for the counted PR130 semantic-allocation receiver."""

from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[3]
TREE = REPO / "src/tac/pr130_runtime/dv1_cpu_runtime"
PROOF_PATH = Path("/Volumes/VertigoDataTier/pact/ddm_sr1_20260809/SR1_RECEIVER_PROOF.json")


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


proof_tool = load_module(
    "ddm_sr1_semantic_alloc_schema_tool",
    REPO / "experiments/ddm_sr1_semantic_alloc_schema.py",
)
inflate = proof_tool.load_runtime("ddm_sr1_test_runtime", TREE)


def test_sd1m_v1_pins_the_exact_allocation_order() -> None:
    template = inflate.SemanticTokenRenderer(96).state_dict()
    quantized_names = tuple(name for name, value in template.items() if value.ndim >= 2)
    assert quantized_names == inflate.SEMANTIC_MIXED_V1_NAMES

    header = b"SD1M\x01\x10" + b"\x44" * 8
    allocation, remaining, format_name = inflate.semantic_allocation(header, template)
    assert format_name == "sd1_mixed_v1"
    assert not remaining
    assert tuple(allocation) == quantized_names
    assert set(allocation.values()) == {4}

    reordered = OrderedDict(reversed(tuple(template.items())))
    with pytest.raises(ValueError, match="template order differs"):
        inflate.semantic_allocation(header, reordered)


def test_unseen_all_q4_sd1m_length_reaches_the_full_loader() -> None:
    reference = load_module(
        "ddm_sr1_test_sd1_reference",
        REPO / "experiments/ddm_sd1_semantic_rd_curve.py",
    )
    template = inflate.SemanticTokenRenderer(96).state_dict()
    allocation = {name: 4 for name, value in template.items() if value.ndim >= 2}
    semantic, expected = reference.pack_semantic_state(
        template,
        allocation,
        legacy_int4=False,
    )
    assert len(semantic) == 40_266
    assert len(semantic) not in inflate.SEMANTIC_WIDTH_BY_PAYLOAD_BYTES

    basis_count = inflate.CARRIER_DIM * 3 * inflate.CARRIER_H * inflate.CARRIER_W
    coeff_count = inflate.N * inflate.CARRIER_DIM
    scale_bytes = inflate.CARRIER_DIM * 4
    carrier = (
        bytes(scale_bytes)
        + bytes((basis_count * 4 + 7) // 8)
        + bytes(scale_bytes)
        + bytes(((coeff_count + 1) // 2) * 3)
    )
    raw = struct.pack("<II", len(semantic), len(carrier)) + semantic + carrier
    model, basis, coeff = inflate.unpack_semantic_pose(raw)
    for name, value in expected.items():
        assert value.equal(model.state_dict()[name]), name
    assert basis.shape == (12, 3, 24, 32)
    assert coeff.shape == (600, 12)


@pytest.mark.skipif(not PROOF_PATH.is_file(), reason="SR1 custody is not mounted")
def test_real_all_q4_record_uses_full_loader_and_matches_legacy_bytes() -> None:
    receipt = json.loads(PROOF_PATH.read_text())
    assert receipt["complete"]
    intended_inflate = receipt["sources"]["extended_inflate"]
    assert proof_tool.sha256_file(Path(intended_inflate["path"])) == intended_inflate["sha256"]

    retained = receipt["retained_payloads"]
    raw = Path(retained["all_q4_record_semantic_pose"]["path"]).read_bytes()
    model, basis, coeff = inflate.unpack_semantic_pose(raw)
    all_q4_state = proof_tool.state_wire(model.state_dict())
    all_q4_carrier = proof_tool.tensor_pair_wire(basis, coeff)

    assert all_q4_state == Path(retained["decoded_extended_all_q4_full_loader_state"]["path"]).read_bytes()
    assert all_q4_state == Path(retained["decoded_legacy_full_loader_state"]["path"]).read_bytes()
    assert all_q4_carrier == Path(retained["decoded_extended_all_q4_carrier"]["path"]).read_bytes()
    assert all_q4_carrier == Path(retained["decoded_legacy_carrier"]["path"]).read_bytes()


@pytest.mark.skipif(not PROOF_PATH.is_file(), reason="SR1 custody is not mounted")
def test_real_receiver_receipt_closes_raw_parity_and_archive_pricing() -> None:
    receipt = json.loads(PROOF_PATH.read_text())
    raw = receipt["raw_byte_identity"]
    assert raw["complete"] and raw["byte_equal"]
    assert raw["bytes_compared"] == 3_662_409_600
    assert raw["sha256"] == ("a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353")
    legacy = raw["legacy_no_record"]["artifact"]
    all_q4 = raw["extended_all_q4_record"]["artifact"]
    assert legacy["bytes"] == all_q4["bytes"]
    assert legacy["sha256"] == all_q4["sha256"]
    assert Path(legacy["path"]).stat().st_size == legacy["bytes"]
    assert Path(all_q4["path"]).stat().st_size == all_q4["bytes"]

    pricing = receipt["pricing"]
    assert pricing["base_archive_bytes"] == 191_052
    assert pricing["selected_archive_bytes"] == 190_204
    assert pricing["honest_net_delta_archive_bytes"] == -848
    assert pricing["allocation_record_raw_bytes"] == 14
    assert pricing["allocation_record_complete_archive_marginal_bytes"] == 0
    assert pricing["schema_is_already_counted_in_selected_archive"]
    assert pricing["subtracting_schema_again_would_double_count"]


def test_runtime_dependency_manifest_pins_landed_inflate() -> None:
    manifest = json.loads((TREE / "runtime-dependencies.json").read_text())
    landed = subprocess.run(
        [
            "git",
            "show",
            "HEAD:src/tac/pr130_runtime/dv1_cpu_runtime/inflate.py",
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    assert manifest["source"]["copied_files"]["inflate.py"] == (proof_tool.sha256_bytes(landed))
