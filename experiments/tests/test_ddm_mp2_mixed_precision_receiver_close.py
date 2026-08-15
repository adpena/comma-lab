from __future__ import annotations

import importlib
from collections import OrderedDict

import numpy as np
import pytest
import torch

builder = importlib.import_module("experiments.ddm_mp2_mixed_precision_receiver_close")
receiver = importlib.import_module("experiments.ddm_mp2_semantic_receiver")


def current_template():
    records, _, _ = builder.mz2._load_records()
    return records, OrderedDict(
        (
            record.schema.name,
            torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32)),
        )
        for record in records
    )


def assert_state_equal(expected, actual):
    assert tuple(actual) == tuple(expected)
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name


def test_receiver_decodes_current_mixed_packet_exactly():
    records, template = current_template()
    payload, expected = builder.mz2._selected_mixed(records)
    actual = receiver.unpack_variant_semantic_or_none(payload, template)
    assert actual is not None
    assert_state_equal(expected, actual)


@pytest.mark.parametrize("keep_percent", [25, 37, 50, 62, 75, 87])
def test_receiver_decodes_every_current_row_prune_packet_exactly(keep_percent):
    _, template = current_template()
    payload, expected, _ = builder.mz2.sm3.pack_prune_candidate(template, keep_percent)
    actual = receiver.unpack_variant_semantic_or_none(payload, template)
    assert actual is not None
    assert_state_equal(expected, actual)


def test_receiver_preserves_legacy_dispatch_and_refuses_unknown_sm3r():
    _, template = current_template()
    assert receiver.unpack_variant_semantic_or_none(b"legacy-q4", template) is None
    with pytest.raises(receiver.MP2SemanticFormatError, match="unsupported SM3R mode"):
        receiver.unpack_variant_semantic_or_none(b"SM3R\x01\x04\x00\x00", template)


def test_runtime_patch_is_additive_and_single_site(tmp_path):
    source = builder.BASE_GENERATION / "cpr1/inflate.py"
    candidate = tmp_path / "inflate.py"
    candidate.write_bytes(source.read_bytes())
    builder.patch_inner_runtime(candidate)
    first_pass = candidate.read_bytes()
    builder.patch_inner_runtime(candidate)
    assert candidate.read_bytes() == first_pass
    patched = candidate.read_text(encoding="utf-8")
    assert patched.count("from ddm_mp2_semantic_receiver import unpack_variant_semantic_or_none") == 1
    assert patched.count("tagged_state = unpack_variant_semantic_or_none") == 1
    assert "tagged_state = unpack_semantic(semantic_blob, semantic.state_dict())" in patched
    assert "SEMANTIC_WIDTH_BY_PAYLOAD_BYTES[semantic_bytes]" in patched

    residual = tmp_path / "residual_archive.py"
    residual.write_bytes((builder.BASE_GENERATION / "runtime/residual_archive.py").read_bytes())
    builder.patch_residual_runtime(residual)
    residual_first = residual.read_bytes()
    builder.patch_residual_runtime(residual)
    assert residual.read_bytes() == residual_first
    assert b"tagged_semantic = semantic_body.startswith" in residual_first

    f26 = tmp_path / "f26_inflate.py"
    f26.write_bytes((builder.BASE_GENERATION / "runtime/f26_inflate.py").read_bytes())
    builder.patch_f26_runtime(f26)
    f26_first = f26.read_bytes()
    builder.patch_f26_runtime(f26)
    assert f26.read_bytes() == f26_first
    assert b"renderer.unpack_variant_semantic_or_none" in f26_first


def test_candidate_set_and_hv1_member_geometry_are_pinned():
    score_result = builder.json.loads(builder.MZ2_SCORE_RESULT.read_text(encoding="utf-8"))
    rows = builder.candidate_rows(score_result)
    assert len(rows) == 7
    member = builder.read_stored_member(builder.BASE_GENERATION / "archive.zip")
    parts = builder.split_member(member)
    assert parts["semantic_size"] == builder.BASE_SEMANTIC_STREAM_BYTES
    assert len(parts["model"]) + len(parts["tail"]) == len(member)
