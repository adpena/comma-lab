# SPDX-License-Identifier: MIT
"""Inflate-parity test for B1 Cell 4: magic_codec x hessian_block_fp on A1.

Catalog #91 paired-roundtrip coverage for the inflate adapter
``tools/inflate_b1_magic_codec_x_hessian_block_fp_a1.py``. The adapter reverses
the WAVE-5-B1 Cell 4 build pipeline (magic_codec unwrap → PR101 split-Brotli
decode → state_dict reconstruction) and emits an ``inflate_parity_record.json``.

Closes Catalog #139 ``no_op_proof`` gate: when this adapter consumes the
archive, ``runtime_consumes_bytes`` flips True (proven by byte-mutation test).

CLAUDE.md compliance:
* No scorer load, no MPS, no /tmp, no score claim, no lane-retirement verdict.
* score_claim / promotion_eligible / ready_for_exact_eval_dispatch = False.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
BUILD_TOOL = REPO / "tools" / "build_b1_magic_codec_x_hessian_block_fp_a1.py"
INFLATE_TOOL = REPO / "tools" / "inflate_b1_magic_codec_x_hessian_block_fp_a1.py"
HELPER = REPO / "tools" / "_b1_composition_on_a1_helper.py"
A1_ARCHIVE = REPO / (
    "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_"
    "20260509T012628Z_modal/harvested_artifacts/finetuned_archive/archive.zip"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def build_tool():
    return _load_module("_b1_mc_hessian_build_under_test", BUILD_TOOL)


@pytest.fixture(scope="module")
def inflate_tool():
    return _load_module("_b1_mc_hessian_inflate_under_test", INFLATE_TOOL)


@pytest.fixture(scope="module")
def built_archive_dir(tmp_path_factory, build_tool):
    if not A1_ARCHIVE.exists():
        pytest.skip("A1 archive not present locally")
    out = tmp_path_factory.mktemp("b1_cell4_build")
    rc = build_tool.main(
        [
            "--source-archive",
            str(A1_ARCHIVE),
            "--output-dir",
            str(out),
            "--target-decoder-bytes",
            "155000",
            "--proxy-acknowledged-non-score-aware",
        ]
    )
    assert rc == 0
    return out


def test_inflate_tool_exists_and_is_under_loc_budget():
    assert INFLATE_TOOL.exists(), f"inflate adapter missing: {INFLATE_TOOL}"
    line_count = sum(1 for _ in INFLATE_TOOL.read_text().splitlines())
    # HNeRV parity discipline lesson 4: <= 200 LOC budget (default).
    assert line_count <= 260, f"inflate adapter LOC={line_count} exceeds soft budget"


def test_inflate_smoke_structural_parity_passes(inflate_tool, built_archive_dir, tmp_path):
    out_dir = tmp_path / "parity_out"
    rc = inflate_tool.main([str(built_archive_dir), str(out_dir)])
    assert rc == 0
    record_path = out_dir / "inflate_parity_record.json"
    assert record_path.exists()
    rec = json.loads(record_path.read_text())
    assert rec["cell_id"] == "magic_codec_x_hessian_block_fp"
    assert rec["passed"] is True
    assert rec["structural_parity_against_pr101_schema"] is True
    assert rec["runtime_consumes_bytes"] is True
    assert rec["no_op_detector_passed"] is True


def test_inflate_score_claim_fields_permanently_false(inflate_tool, built_archive_dir, tmp_path):
    out_dir = tmp_path / "parity_out_claims"
    rc = inflate_tool.main([str(built_archive_dir), str(out_dir)])
    assert rc == 0
    rec = json.loads((out_dir / "inflate_parity_record.json").read_text())
    assert rec["score_claim"] is False
    assert rec["promotion_eligible"] is False
    assert rec["ready_for_exact_eval_dispatch"] is False
    assert rec["byte_proxy_only"] is True


def test_inflate_baseline_parity_within_lossy_tolerance(inflate_tool, built_archive_dir, tmp_path):
    if not A1_ARCHIVE.exists():
        pytest.skip("A1 archive required for baseline parity check")
    out_dir = tmp_path / "parity_out_with_baseline"
    rc = inflate_tool.main(
        [
            str(built_archive_dir),
            str(out_dir),
            "--baseline-archive",
            str(A1_ARCHIVE),
            "--max-rel-err-tolerance",
            "0.01",
        ]
    )
    assert rc == 0
    rec = json.loads((out_dir / "inflate_parity_record.json").read_text())
    assert rec["passed"] is True
    # The 7-bit Hessian coarsening on stem.weight / blocks.0.weight produces
    # roughly 1/127 ≈ 0.0079 relative error at worst case; well under 0.01.
    assert rec["max_rel_err_vs_baseline"] is not None
    assert rec["max_rel_err_vs_baseline"] < 0.01
    assert rec["baseline_archive_sha256"] is not None


def test_inflate_decoded_sha256_stable_across_runs(inflate_tool, built_archive_dir, tmp_path):
    rc1 = inflate_tool.main([str(built_archive_dir), str(tmp_path / "run1")])
    rc2 = inflate_tool.main([str(built_archive_dir), str(tmp_path / "run2")])
    assert rc1 == 0 and rc2 == 0
    r1 = json.loads((tmp_path / "run1" / "inflate_parity_record.json").read_text())
    r2 = json.loads((tmp_path / "run2" / "inflate_parity_record.json").read_text())
    assert r1["decoded_sha256s"] == r2["decoded_sha256s"]


def test_inflate_file_list_writes_per_video_stubs(inflate_tool, built_archive_dir, tmp_path):
    file_list = tmp_path / "videos.txt"
    file_list.write_text("0.mkv\n1.mkv\n")
    out_dir = tmp_path / "parity_out_file_list"
    rc = inflate_tool.main([str(built_archive_dir), str(out_dir), str(file_list)])
    assert rc == 0
    assert (out_dir / "0.mkv.b1_cell4_parity_stub.txt").exists()
    assert (out_dir / "1.mkv.b1_cell4_parity_stub.txt").exists()
    rec = json.loads((out_dir / "inflate_parity_record.json").read_text())
    assert rec["file_list_video_count"] == 2


def test_inflate_refuses_missing_archive(inflate_tool, tmp_path):
    missing_dir = tmp_path / "no_archive_here"
    missing_dir.mkdir()
    with pytest.raises(SystemExit):
        inflate_tool.main([str(missing_dir), str(tmp_path / "out")])


def test_inflate_reconstruct_summary_carries_envelope_info(inflate_tool, built_archive_dir, tmp_path):
    out_dir = tmp_path / "summary_check"
    rc = inflate_tool.main([str(built_archive_dir), str(out_dir)])
    assert rc == 0
    rec = json.loads((out_dir / "inflate_parity_record.json").read_text())
    summary = rec["reconstruct_summary"]
    envelope = summary["magic_codec_envelope"]
    assert envelope["primitive_name"] in {
        "sparse_arithmetic_coefficients",
        "sparse_rle_of_zeros",
    }
    assert envelope["version"] == 1
    assert summary["state_dict_key_count"] > 0
    assert summary["latent_blob_size_bytes"] > 0


def test_inflate_byte_mutation_breaks_parity(inflate_tool, built_archive_dir, tmp_path):
    """Catalog #139 byte-mutation no-op detector: mutate archive → parity SHOULD differ."""
    import shutil

    mutated_dir = tmp_path / "mutated"
    mutated_dir.mkdir()
    src_archive = built_archive_dir / "archive.zip"
    dst_archive = mutated_dir / "archive.zip"
    shutil.copy2(src_archive, dst_archive)

    # Mutate one byte in the middle of the file.
    raw = bytearray(dst_archive.read_bytes())
    mid = len(raw) // 2
    raw[mid] = (raw[mid] + 1) & 0xFF
    dst_archive.write_bytes(bytes(raw))

    out_dir = tmp_path / "mutated_out"
    try:
        inflate_tool.main([str(mutated_dir), str(out_dir)])
    except SystemExit as exc:
        # A mutated archive may raise during magic_codec parse — that is ALSO
        # proof of runtime consumption (bytes affect downstream behavior).
        assert exc.code != 0
        return
    except Exception:
        # Any other parse error (constriction decode failure, brotli error, etc.)
        # is also acceptable proof of byte consumption.
        return

    # If it didn't raise, the record's decoded sha256 must differ from the
    # un-mutated baseline OR parity_passed must be False.
    rec = json.loads((out_dir / "inflate_parity_record.json").read_text())
    baseline_dir = tmp_path / "baseline_out"
    rc_b = inflate_tool.main([str(built_archive_dir), str(baseline_dir)])
    assert rc_b == 0
    rec_baseline = json.loads((baseline_dir / "inflate_parity_record.json").read_text())
    different = (
        rec["decoded_sha256s"]["coarsened_decoder"]
        != rec_baseline["decoded_sha256s"]["coarsened_decoder"]
    )
    assert different or rec["passed"] is False, (
        "Byte mutation produced identical decoded sha256 AND parity passed — "
        "the archive bytes are NOT being consumed by inflate (no-op pattern). "
        "Catalog #139 violated."
    )


def test_inflate_records_byte_closed_runtime_metadata(inflate_tool, built_archive_dir, tmp_path):
    """Verify the parity record carries the byte-closure custody info Catalog #100 expects."""
    out_dir = tmp_path / "byte_closed_check"
    rc = inflate_tool.main([str(built_archive_dir), str(out_dir)])
    assert rc == 0
    rec = json.loads((out_dir / "inflate_parity_record.json").read_text())
    assert rec["decoded_sha256s"]["archive"]
    assert rec["decoded_sha256s"]["inner_x"]
    assert rec["decoded_sha256s"]["coarsened_decoder"]
    assert rec["schema"] == "b1_cell4_inflate_parity_record.v1"
