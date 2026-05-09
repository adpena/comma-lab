"""Tests for :mod:`tac.codec_op_admm_adapter`.

The bridge is planning-only: it may feed CodecOp byte/RMS evidence into
Joint-ADMM allocation logic, but it must fail closed on decode coverage and
must never mark rows as score claims or dispatch-ready.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tac.codec_op_admm_adapter import (
    CodecOpADMMAdapter,
    CodecOpADMMAdapterError,
    adapt_codec_op_class,
    build_codec_op_tensor_contract,
    codec_op_to_admm_planning_row,
)
from tac.joint_admm_coordinator import StreamProximalCodec


@dataclass
class _FakeEncodeResult:
    blob: bytes
    bytes_in: int
    bytes_out: int
    op_name: str
    op_state: dict[str, Any] = field(default_factory=dict)


class _IdentityCodecOp:
    name = "identity_codec_op"

    def __init__(self, *, quality: int = 1) -> None:
        self.quality = int(quality)

    def validate(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> SimpleNamespace:
        findings: list[str] = []
        if self.quality < 1:
            findings.append("quality must be >= 1")
        if not state_dict:
            findings.append("empty state_dict")
        return SimpleNamespace(passed=not findings, findings=findings)

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> _FakeEncodeResult:
        payload = {
            key: {
                "shape": [int(dim) for dim in tensor.shape],
                "values": tensor.detach().cpu().reshape(-1).tolist(),
            }
            for key, tensor in sorted(state_dict.items())
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return _FakeEncodeResult(
            blob=blob,
            bytes_in=sum(int(t.numel() * t.element_size()) for t in state_dict.values()),
            bytes_out=len(blob),
            op_name=self.name,
            op_state={"quality": self.quality},
        )

    def decode(
        self,
        blob: bytes,
        *,
        op_state: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        del op_state, context
        payload = json.loads(blob.decode("utf-8"))
        return {
            key: torch.tensor(entry["values"], dtype=torch.float32).reshape(entry["shape"])
            for key, entry in payload.items()
        }


class _MissingDecodeCodecOp(_IdentityCodecOp):
    name = "missing_decode_codec_op"

    def decode(
        self,
        blob: bytes,
        *,
        op_state: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        decoded = super().decode(blob, op_state=op_state, context=context)
        decoded.pop("b.bias")
        return decoded


class _ShapeMismatchCodecOp(_IdentityCodecOp):
    name = "shape_mismatch_codec_op"

    def decode(
        self,
        blob: bytes,
        *,
        op_state: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        decoded = super().decode(blob, op_state=op_state, context=context)
        decoded["a.weight"] = decoded["a.weight"].reshape(-1)
        return decoded


class _DtypeMismatchCodecOp(_IdentityCodecOp):
    name = "dtype_mismatch_codec_op"

    def decode(
        self,
        blob: bytes,
        *,
        op_state: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        decoded = super().decode(blob, op_state=op_state, context=context)
        decoded["a.weight"] = decoded["a.weight"].to(torch.float64)
        return decoded


def _state_dict() -> dict[str, torch.Tensor]:
    return {
        "a.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "b.bias": torch.tensor([0.25, -0.5, 0.75], dtype=torch.float32),
    }


def test_adapter_is_stream_proximal_codec_and_emits_fail_closed_state() -> None:
    state_dict = _state_dict()
    op = _IdentityCodecOp(quality=3)
    encoded = op.encode(state_dict, context={})
    decoded = op.decode(encoded.blob, op_state=encoded.op_state, context={})
    for key, tensor in state_dict.items():
        assert torch.allclose(decoded[key], tensor)

    adapter = adapt_codec_op_class(
        _IdentityCodecOp,
        state_dict,
        score_delta=0.125,
        marginal=0.01,
        op_params={"quality": 3},
        source_label="unit_test_codec_op",
    )

    assert isinstance(adapter, StreamProximalCodec)
    step = adapter.proximal_step(target_bytes=10_000, dual=0.5)
    row = step.state

    assert step.encoded_bytes == row["bytes_out"]
    assert step.score_delta == 0.125
    assert step.marginal == 0.01
    assert row["schema"] == "codec_op_admm_adapter_planning_row_v1"
    assert row["stream_name"] == "codec_op:identity_codec_op"
    assert row["decode_coverage_status"] == "full"
    assert row["score_claim"] is False
    assert row["dispatchable"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["score_affecting_payload_changed"] is False
    assert row["charged_bits_changed"] is False
    assert row["exact_cuda_auth_eval"] is False
    assert row["archive_sha256"] is None
    assert row["archive_bytes"] is None
    assert "missing_exact_cuda_auth_eval" in row["dispatch_blockers"]
    assert row["admm_query"] == {
        "target_bytes": 10000.0,
        "dual": 0.5,
        "fixed_codec_op_bytes_out": row["bytes_out"],
        "target_bytes_exceeded": False,
    }


def test_planning_row_records_deterministic_tensor_contract_and_custody() -> None:
    adapter = CodecOpADMMAdapter(
        _IdentityCodecOp(quality=2),
        _state_dict(),
        score_delta=0.0,
        marginal=0.02,
        op_params={"quality": 2},
        context={"purpose": "test"},
    )

    row_a = adapter.to_planning_row()
    row_b = adapter.to_planning_row()

    assert row_a == row_b
    assert row_a["op_params"] == {"quality": 2}
    assert row_a["context_keys"] == ["purpose"]
    assert row_a["op_state_keys"] == ["quality"]
    assert len(row_a["blob_sha256"]) == 64
    assert len(row_a["op_state_sha256"]) == 64
    assert len(row_a["tensor_contract_sha256"]) == 64
    assert [entry["key"] for entry in row_a["tensor_contract"]] == [
        "a.weight",
        "b.bias",
    ]
    assert all(len(entry["sha256"]) == 64 for entry in row_a["tensor_contract"])
    assert row_a["decode_validation"]["matched_tensor_keys"] == [
        "a.weight",
        "b.bias",
    ]
    assert row_a["decode_validation"]["shape_mismatch_tensor_keys"] == []
    assert "materialized_payload_path" not in row_a
    assert "materialized_payload_bytes" not in row_a
    assert "materialized_payload_sha256" not in row_a
    assert "materialized_payload_contract" not in row_a


def test_planning_row_records_real_materialized_payload_file(tmp_path: Path) -> None:
    state_dict = _state_dict()
    op = _IdentityCodecOp(quality=2)
    result = op.encode(state_dict, context={})
    payload_path = tmp_path / "eval_00000.section"
    payload_path.write_bytes(result.blob)

    adapter = CodecOpADMMAdapter(
        op,
        state_dict,
        score_delta=0.0,
        marginal=0.02,
        op_params={"quality": 2},
        materialized_payload_path=payload_path,
        materialized_payload_contract="pr106_decoder_packed_brotli",
    )

    row = adapter.to_planning_row()

    assert row["materialized_payload_path"] == payload_path.as_posix()
    assert row["materialized_payload_bytes"] == len(result.blob)
    assert row["materialized_payload_sha256"] == hashlib.sha256(
        result.blob
    ).hexdigest()
    assert row["materialized_payload_contract"] == "pr106_decoder_packed_brotli"
    assert row["score_claim"] is False
    assert row["dispatchable"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["promotion_eligible"] is False


def test_materialized_payload_path_must_exist() -> None:
    adapter = CodecOpADMMAdapter(
        _IdentityCodecOp(),
        _state_dict(),
        score_delta=0.0,
        marginal=0.0,
        materialized_payload_path=Path("does-not-exist.section"),
    )

    with pytest.raises(CodecOpADMMAdapterError, match="existing file"):
        adapter.to_planning_row()


def test_materialized_payload_file_must_match_codec_op_blob(tmp_path: Path) -> None:
    payload_path = tmp_path / "mismatch.section"
    payload_path.write_bytes(b"not-the-codec-op-blob")

    adapter = CodecOpADMMAdapter(
        _IdentityCodecOp(),
        _state_dict(),
        score_delta=0.0,
        marginal=0.0,
        materialized_payload_path=payload_path,
    )

    with pytest.raises(CodecOpADMMAdapterError, match="do not match CodecOp blob"):
        adapter.to_planning_row()


def test_tensor_contract_helper_is_sorted_and_stable() -> None:
    contract_a = build_codec_op_tensor_contract(_state_dict())
    contract_b = build_codec_op_tensor_contract(
        {"b.bias": _state_dict()["b.bias"], "a.weight": _state_dict()["a.weight"]}
    )

    assert contract_a == contract_b
    assert [entry["key"] for entry in contract_a] == ["a.weight", "b.bias"]
    assert contract_a[0]["shape"] == [2, 3]
    assert contract_a[0]["dtype"] == "torch.float32"
    assert contract_a[0]["nbytes"] == 24


def test_codec_op_to_planning_row_helper_keeps_dispatch_flags_false() -> None:
    row = codec_op_to_admm_planning_row(
        _IdentityCodecOp(quality=4),
        _state_dict(),
        score_delta=0.0,
        marginal=0.03,
        op_params={"quality": 4},
        stream_name="custom_stream",
    )

    assert row["stream_name"] == "custom_stream"
    assert row["op_class"] == "_IdentityCodecOp"
    assert row["score_claim"] is False
    assert row["dispatchable"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["planning_objectives"]["decode_coverage_status"] == "full"


def test_missing_full_decode_fails_closed() -> None:
    adapter = CodecOpADMMAdapter(
        _MissingDecodeCodecOp(),
        _state_dict(),
        score_delta=0.0,
        marginal=0.0,
    )

    with pytest.raises(CodecOpADMMAdapterError, match="full decode/shape validation"):
        adapter.to_planning_row()


def test_shape_mismatch_fails_closed() -> None:
    adapter = CodecOpADMMAdapter(
        _ShapeMismatchCodecOp(),
        _state_dict(),
        score_delta=0.0,
        marginal=0.0,
    )

    with pytest.raises(CodecOpADMMAdapterError, match="shape_mismatch"):
        adapter.proximal_step(target_bytes=10_000, dual=0.0)


def test_dtype_mismatch_fails_closed() -> None:
    adapter = CodecOpADMMAdapter(
        _DtypeMismatchCodecOp(),
        _state_dict(),
        score_delta=0.0,
        marginal=0.0,
    )

    with pytest.raises(CodecOpADMMAdapterError, match="dtype_mismatch"):
        adapter.to_planning_row()


def test_bad_cached_marginal_rejected_before_admm_use() -> None:
    with pytest.raises(CodecOpADMMAdapterError, match="marginal must be >= 0"):
        CodecOpADMMAdapter(
            _IdentityCodecOp(),
            _state_dict(),
            score_delta=0.0,
            marginal=-1e-3,
        )


def test_validate_failure_fails_closed() -> None:
    adapter = CodecOpADMMAdapter(
        _IdentityCodecOp(quality=0),
        _state_dict(),
        score_delta=0.0,
        marginal=0.0,
    )

    with pytest.raises(CodecOpADMMAdapterError, match="validate\\(\\) failed"):
        adapter.to_planning_row()
