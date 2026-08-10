from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from experiments import ddm_cp2_composition_receiver_and_harness as cp2
from experiments import ddm_sm3_semantic_representation as sm3
from experiments.ddm_cp2_runtime import sm3r_receiver


def _require_real_inputs() -> None:
    required = [
        cp2.BASE_ARCHIVE,
        sm3.DEFAULT_CHECKPOINT,
        cp2.TOKEN_CANDIDATES["temporal_reversion"].path,
        cp2.SEMANTIC_CANDIDATES["vector_scale_vq32"].path,
        cp2.SEMANTIC_CANDIDATES["pointwise_lowrank_r32"].path,
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("DDM-CP2 retained PR130/AI1/SM3 custody is not mounted")


@pytest.fixture(scope="module")
def real_template() -> OrderedDict[str, torch.Tensor]:
    _require_real_inputs()
    checkpoint = torch.load(
        sm3.DEFAULT_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    return OrderedDict(checkpoint["state_dict"])


def test_sm3r_three_cases_on_real_retained_fields(
    real_template: OrderedDict[str, torch.Tensor],
) -> None:
    base = sm3.sd1.read_base_archive(cp2.BASE_ARCHIVE)
    assert sm3r_receiver.unpack_sm3r_or_none(base.semantic_blob, real_template) is None

    for candidate_id in ("vector_scale_vq32", "pointwise_lowrank_r32"):
        spec = cp2.SEMANTIC_CANDIDATES[candidate_id]
        assert spec is not None
        blob = spec.path.read_bytes()
        expected = sm3.unpack_candidate(blob, real_template)
        actual = sm3r_receiver.unpack_sm3r_or_none(blob, real_template)
        assert actual is not None
        assert list(actual) == list(expected)
        assert all(torch.equal(actual[name], expected[name]) for name in expected)

    known = cp2.SEMANTIC_CANDIDATES["vector_scale_vq32"]
    assert known is not None
    unknown_version = bytearray(known.path.read_bytes())
    unknown_version[4] = 2
    with pytest.raises(sm3r_receiver.SM3RFormatError, match="version"):
        sm3r_receiver.unpack_sm3r_or_none(bytes(unknown_version), real_template)

    unknown_mode = bytearray(known.path.read_bytes())
    unknown_mode[5] = 99
    with pytest.raises(sm3r_receiver.SM3RFormatError, match="mode"):
        sm3r_receiver.unpack_sm3r_or_none(bytes(unknown_mode), real_template)

    with pytest.raises(sm3r_receiver.SM3RFormatError, match="truncated"):
        sm3r_receiver.unpack_sm3r_or_none(sm3r_receiver.MAGIC, real_template)
    with pytest.raises(sm3r_receiver.SM3RFormatError, match="trailing"):
        sm3r_receiver.unpack_sm3r_or_none(known.path.read_bytes() + b"\x00", real_template)

    row_prune = cp2.SM3_ROOT / "film_row_prune_keep75/semantic.bin"
    if row_prune.is_file():
        with pytest.raises(sm3r_receiver.SM3RFormatError, match="mode"):
            sm3r_receiver.unpack_sm3r_or_none(row_prune.read_bytes(), real_template)


def test_pins_cover_every_chartered_candidate() -> None:
    _require_real_inputs()
    for candidates in (cp2.TOKEN_CANDIDATES, cp2.SEMANTIC_CANDIDATES):
        for candidate_id, spec in candidates.items():
            if spec is None:
                continue
            cp2.require_record(
                spec.path,
                size=spec.bytes,
                digest=spec.sha256,
                label=candidate_id,
            )


def test_real_temporal_driver_rebuild_is_byte_identical_and_retained() -> None:
    _require_real_inputs()
    output = cp2.DEFAULT_OUTPUT_ROOT / "test_temporal_inherit_control"
    result = cp2.build(
        output,
        semantic_id="inherit",
        token_id="temporal_reversion",
        resume_from=output / "resume.json",
        minimum_free_bytes=1 << 20,
    )
    assert result["complete"] is True
    assert result["checks"]["driver_rebuild_byte_identical"] is True
    assert result["checks"]["archive_double_build_byte_identical"] is True
    assert result["actual_archive_bytes"] == 188_636
    assert result["actual_archive_delta_bytes"] == -2_416
    assert result["interaction_gap_bytes_actual_minus_additive"] == 0
    assert result["retained"]["archive"]["sha256"] == cp2.TOKEN_CANDIDATES["temporal_reversion"].sha256
    legacy = cp2.SEMANTIC_CANDIDATES["legacy_q4"]
    assert legacy is not None
    assert result["retained"]["semantic_state"]["sha256"] == legacy.expected_state_sha256


def test_retained_composed_build_and_real_inflate_receipts_when_present() -> None:
    leader = cp2.DEFAULT_OUTPUT_ROOT / "pointwise_lowrank_r32__temporal_reversion"
    build_path = leader / "build_receipt.json"
    if not build_path.is_file():
        pytest.skip("DDM-CP2 composed leader has not been materialized yet")
    build = json.loads(build_path.read_text())
    assert build["complete"] is True
    assert build["checks"]["archive_double_build_byte_identical"] is True
    assert build["checks"]["outer_receiver_models_raw_byte_identical"] is True
    assert build["checks"]["shipped_sm3r_state_equals_packer_state"] is True
    for record in build["retained"].values():
        cp2.require_record(
            Path(record["path"]),
            size=record["bytes"],
            digest=record["sha256"],
            label="retained composed artifact",
        )

    inflate_path = leader / "receiver_parseback/inflate_receipt.json"
    if not inflate_path.is_file():
        pytest.skip("DDM-CP2 real inflate.sh parse-back has not completed yet")
    inflate = json.loads(inflate_path.read_text())
    assert inflate["complete"] is True
    assert inflate["returncode"] == 0
    assert inflate["within_1800_second_inflate_limit"] is True
    assert inflate["raw"]["bytes"] == 3_662_409_600
