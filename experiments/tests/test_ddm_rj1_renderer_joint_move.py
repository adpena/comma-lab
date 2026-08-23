from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from experiments import ddm_rj1_renderer_joint_move as rj1
from experiments import ddm_wd2_student_receiver as receiver
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def _source_state() -> OrderedDict[str, torch.Tensor]:
    spec = receiver.StudentSpec("source", "dense", 96, 4)
    return OrderedDict(receiver.StudentSemanticRenderer(spec).state_dict())


def test_rungs_are_three_distinct_non_depth_representation_forms() -> None:
    rows = rj1.representation_rungs(_source_state())
    assert [spec.form for spec, _, _ in rows] == ["dense", "factorized", "flattened"]
    assert {spec.depth for spec, _, _ in rows} == {4}
    assert len({description for _, _, description in rows}) == 3


def test_factorized_full_rank_reconstructs_pointwise_operator() -> None:
    torch.manual_seed(0)
    matrix = torch.randn(16, 16)
    u, singular, vh = rj1._canonical_svd(matrix)
    reconstructed = (u * singular[None, :]) @ vh
    torch.testing.assert_close(reconstructed, matrix, rtol=1e-5, atol=1e-5)


def test_factorized_rung_uses_real_svd_factors_and_original_bias() -> None:
    source = _source_state()
    result = rj1.build_factorized_state(source, rank=32)
    assert result["blocks.0.down.weight"].shape == (32, 96, 1, 1)
    assert result["blocks.0.up.weight"].shape == (96, 32, 1, 1)
    assert torch.count_nonzero(result["blocks.0.down.weight"]) > 0
    assert torch.count_nonzero(result["blocks.0.up.weight"]) > 0
    torch.testing.assert_close(result["blocks.0.up.bias"], source["blocks.0.pw.bias"])


def test_flattened_rung_amortizes_all_four_film_maps() -> None:
    source = _source_state()
    for block in range(4):
        source[f"blocks.{block}.film.weight"].fill_(float(block + 1))
        source[f"blocks.{block}.film.bias"].fill_(float(10 + block))
    result = rj1.build_flattened_state(source)
    torch.testing.assert_close(result["flat_film.weight"], torch.full_like(result["flat_film.weight"], 2.5))
    torch.testing.assert_close(result["flat_film.bias"], torch.full_like(result["flat_film.bias"], 11.5))


def test_deterministic_zip_repeat_is_byte_identical() -> None:
    member = b"RX1M" + bytes(range(64))
    assert rj1.deterministic_zip(member) == rj1.deterministic_zip(member)


def test_completed_receipt_custody_detects_payload_tamper(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"retained")
    receipt = {"nested": [rj1.file_record(payload)]}
    rj1.verify_retained_file_records(receipt)
    payload.write_bytes(b"tampered")
    with pytest.raises(rj1.RJ1Error, match="custody changed"):
        rj1.verify_retained_file_records(receipt)


def test_rj1_python_files_pass_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=Path.cwd(),
        strict=False,
        roots=(
            "experiments/ddm_rj1_renderer_joint_move.py",
            "experiments/tests/test_ddm_rj1_renderer_joint_move.py",
        ),
    )
    assert findings == []
