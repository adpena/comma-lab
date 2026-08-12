from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_t1r1_container_build_rehearsal as t1r1
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def synthetic_carrier_selector() -> bytes:
    bit_counts = b"\x01\x00\x00\x01\x00\x00"
    scales = bytes(range(96))
    factors = (np.arange(12, dtype=np.int16) + 100).astype("<i2").tobytes()
    biases = np.arange(-6, 6, dtype=np.int8).tobytes()
    lengths = np.tile(np.arange(8, dtype=np.uint8), 4).tobytes()
    ks = np.tile(np.asarray([8, 9], dtype=np.uint8), 6).tobytes()
    return bit_counts + scales + factors + biases + lengths + ks + b"payload-and-selector"


def test_dynamic_cap1_metadata_round_trip() -> None:
    source = synthetic_carrier_selector()
    packed, report = t1r1.pack_dynamic_cap1_metadata(source)
    assert len(packed) == len(source) - 40
    assert report["raw_delta_bytes"] == -40
    assert t1r1.unpack_dynamic_cap1_metadata(packed) == source


def test_dynamic_cap1_metadata_refuses_receiver_incompatible_bias() -> None:
    source = bytearray(synthetic_carrier_selector())
    source[126] = np.asarray([-17], dtype=np.int8).view(np.uint8)[0]
    with pytest.raises(RuntimeError, match="packed domains"):
        t1r1.pack_dynamic_cap1_metadata(bytes(source))


def test_generalized_receiver_source_is_valid_and_tagged() -> None:
    source_path = t1r1.CP135_RUNTIME / "runtime/residual_archive.py"
    if not source_path.is_file():
        return
    adapted = t1r1.generalize_residual_archive_source(source_path.read_text())
    assert "PACKED_CAP1_LENGTH_FLAG = 1 << 15" in adapted
    assert "packed_cap1 = bool" in adapted
    assert "flag is legal only on section three" not in adapted
    compile(adapted, "t1r1_residual_archive.py", "exec")


def test_t1r1_python_files_pass_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=Path.cwd(),
        strict=False,
        roots=(
            "experiments/ddm_t1r1_container_build_rehearsal.py",
            "src/tac/tests/test_ddm_t1r1_container_build_rehearsal.py",
        ),
    )
    assert findings == []


def test_parser_defaults_to_apdatastore_retained_root() -> None:
    args = t1r1.parser().parse_args(["prepare"])
    assert t1r1.retained_root(args.output) == Path("/Volumes/APDataStore/pact/ddm_t1r1/retained")


def test_shipped_receiver_rc64_backend_compiles_into_retained_output(tmp_path: Path) -> None:
    if not t1r1.CP135_RUNTIME.is_dir():
        return
    receipt = t1r1.compile_receiver_rc64(t1r1.CP135_RUNTIME, tmp_path)
    assert receipt["returncode"] == 0
    assert Path(receipt["library"]["path"]).is_file()
    assert receipt["source"]["sha256"] == "05839d1416e68a49c8022d0cccb1581c3e4338fb14c867fc6c116e203c412996"


def test_retained_tree_manifest_excludes_self_and_filesystem_metadata(tmp_path: Path) -> None:
    (tmp_path / "payload.bin").write_bytes(b"payload")
    (tmp_path / "99_TREE_MANIFEST.json").write_text("old")
    (tmp_path / "._payload.bin").write_bytes(b"metadata")
    record = t1r1.retained_tree_record(tmp_path)
    assert [row["relative_path"] for row in record["files"]] == ["payload.bin"]
    assert record["excluded_relative_paths"] == ["99_TREE_MANIFEST.json"]
