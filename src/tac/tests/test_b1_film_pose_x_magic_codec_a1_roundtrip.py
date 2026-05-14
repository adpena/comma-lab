# SPDX-License-Identifier: MIT
"""Roundtrip + structural test for B1 cell: film_pose_x_magic_codec on A1.

Per CLAUDE.md Catalog #91 (`check_encoder_decoder_dequantization_roundtrip_tested`):
every tool that quantises + emits an archive must have a paired roundtrip
test. The builder file declares `# ROUNDTRIP_TESTED:<this file>` to satisfy
the same-line waiver / sibling-test detection.

The roundtrip checks:
* Magic-codec wrapping the A1 decoder bytes is byte-faithful (encode ->
  decode -> equals source).
* The emitted archive has expected zip members (`x` + `FILM` slot).
* The runtime_manifest carries Catalog #100 byte-closure fields with score
  claims permanently False.
* The no-op proof JSON honestly reports `runtime_consumes_bytes: False`
  (no inflate adapter exists yet).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "build_b1_film_pose_x_magic_codec_a1.py"
HELPER = REPO / "tools" / "_b1_composition_on_a1_helper.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def helper():
    return _load_module("_b1_helper_under_test_film_magic", HELPER)


@pytest.fixture(scope="module")
def tool():
    return _load_module("_b1_film_magic_under_test", TOOL)


def test_helper_apply_magic_codec_is_byte_faithful_roundtrip(helper):
    """ENCODE_INFLATE_ROUNDTRIP: magic_codec wrap then unwrap is identity."""
    from tac.packet_compiler.magic_codec import decode_magic_codec

    rng = np.random.default_rng(42)
    payload = rng.integers(-127, 127, size=4096, dtype=np.int8).tobytes()
    result, meta = helper.apply_magic_codec_to_decoder_blob(
        payload, stream_type="weight_tensor", quantize_bits=8
    )
    decoded = decode_magic_codec(result.payload)
    decoded_bytes = decoded.astype(np.int8).tobytes()
    assert decoded_bytes == payload
    assert meta["selected_primitive"] == result.selected_primitive


def test_emit_archive_has_expected_zip_members(helper, tmp_path):
    """The cell's archive must have x + FILM members."""
    inner = b"\x04\x00\x00\x00" + b"\x10" * 16  # minimal valid wire format
    film_slot = helper.build_film_pose_reserved_slot(64)
    archive_path, sha, n_bytes = helper.emit_b1_archive(
        cell_id="test_film_magic",
        out_dir=tmp_path,
        inner_payload=inner,
        film_pose_slot=film_slot,
        composition_metadata={"cell_id": "test_film_magic"},
    )
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        assert "x" in names
        assert "FILM" in names
        assert zf.read("x") == inner
        assert zf.read("FILM") == film_slot
    assert sha == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert n_bytes > 0


def test_film_slot_carries_magic(helper):
    slot = helper.build_film_pose_reserved_slot(4096)
    assert len(slot) == 4096
    assert slot.startswith(helper.FILM_POSE_RESERVED_SLOT_MAGIC)


def test_film_slot_too_small_raises(helper):
    with pytest.raises(ValueError):
        helper.build_film_pose_reserved_slot(2)


def test_runtime_manifest_carries_catalog_100_fields(helper, tmp_path):
    archive = tmp_path / "x.zip"
    archive.write_bytes(b"x")
    rm = helper.build_runtime_manifest(
        cell_id="film_pose_x_magic_codec",
        archive_path=archive,
        archive_sha256="deadbeef" * 8,
        archive_size_bytes=1,
        source_archive_path=archive,
        source_archive_sha256="cafebabe" * 8,
        source_archive_size_bytes=2,
        parser_sections=[{"name": "x"}],
        composition_steps=["a"],
    )
    assert rm["score_claim"] is False
    assert rm["promotion_eligible"] is False
    assert rm["ready_for_exact_eval_dispatch"] is False
    assert rm["byte_proxy_only"] is True
    assert rm["cuda_eval_worth_testing"] is False
    assert "packet_local_inflate_parity_not_run" in rm["dispatch_blockers"]
    assert "no_op_proof_not_run" in rm["dispatch_blockers"]
    assert rm["archive_path"] == str(archive)
    assert rm["parser_section_manifest"] == [{"name": "x"}]


def test_predicted_band_for_cell(helper):
    band = helper.predicted_band_for_cell("film_pose_x_magic_codec")
    assert band[0] < band[1]
    assert band == (3000, 5000)


def test_end_to_end_build_against_real_a1_archive(tool, tmp_path):
    """End-to-end: invoke the tool against the real A1 archive."""
    a1_path = REPO / (
        "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_"
        "20260509T012628Z_modal/harvested_artifacts/finetuned_archive/archive.zip"
    )
    if not a1_path.exists():
        pytest.skip("A1 archive not present")
    out_dir = tmp_path / "b1_film_magic_e2e"
    rc = tool.main(
        [
            "--source-archive",
            str(a1_path),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    archive = out_dir / "archive.zip"
    rm = json.loads((out_dir / "runtime_manifest.json").read_text())
    sm = json.loads((out_dir / "selection_manifest.json").read_text())
    proof = json.loads((out_dir / "no_op_proof.json").read_text())
    assert archive.exists()
    assert rm["cell_id"] == "film_pose_x_magic_codec"
    assert rm["score_claim"] is False
    assert sm["empirical_archive_bytes"] == archive.stat().st_size
    assert proof["runtime_consumes_bytes"] is False  # honest


def test_tool_refuses_tmp_output_dir(tool):
    """CLAUDE.md non-negotiable: no /tmp paths."""
    with tempfile.TemporaryDirectory(), pytest.raises(SystemExit):
        tool.main(
            [
                "--output-dir",
                "/tmp/forbidden_b1_cell_output",
            ]
        )
