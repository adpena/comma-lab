from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import brotli
import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_pr79_pr65_postprocess_atom_candidates.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_pr79_pr65_postprocess_atom_candidates_test", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stored_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            zf.writestr(info, data)


def _fixture_pr65_archive(path: Path, *, active_pair: int = 7) -> Path:
    post = np.zeros((4, 600), dtype=np.uint8)
    post[0, active_pair] = 2
    bias = np.full(600, 13, dtype=np.uint8)
    bias[active_pair] = 14
    region = np.zeros(600, dtype=np.uint8)
    region[active_pair] = 3
    streams = {
        "post": brotli.compress(post.tobytes(), quality=11),
        "shift": brotli.compress(b"SH4" + np.full(600, 40, dtype=np.uint8).tobytes(), quality=11),
        "frac": brotli.compress(b"FH1" + np.full(600, 4, dtype=np.uint8).tobytes(), quality=11),
        "frac2": brotli.compress(b"FH2" + np.full(600, 4, dtype=np.uint8).tobytes(), quality=11),
        "frac3": brotli.compress(b"FH3" + np.full(600, 4, dtype=np.uint8).tobytes(), quality=11),
        "bias": brotli.compress(b"BH1" + bias.tobytes(), quality=11),
        "region": brotli.compress(b"RH1" + region.tobytes(), quality=11),
        "randmulti": b"rand",
    }
    core_lengths = [1001, 1002, 101]
    qpost_lengths = [len(streams[name]) for name in ("post", "shift", "frac", "frac2", "frac3", "bias", "region")]
    header = b"".join(int(n).to_bytes(3, "little") for n in [*core_lengths, *qpost_lengths])
    core = b"a" * core_lengths[0] + b"b" * core_lengths[1] + b"c" * core_lengths[2]
    payload = header + core + b"".join(
        streams[name] for name in ("post", "shift", "frac", "frac2", "frac3", "bias", "region")
    ) + streams["randmulti"]
    _stored_zip(path, {"x": payload})
    return path


def _selected_raw(path: Path, *, height: int = 2, width: int = 3) -> Path:
    arr = np.zeros((1, 2, height, width, 3), dtype=np.uint8)
    path.write_bytes(arr.tobytes())
    return path


def test_pr79_pr65_builder_writes_archive_closed_raw_delta_proof(tmp_path: Path) -> None:
    module = _load()
    source = tmp_path / "source.zip"
    _stored_zip(source, {"p": b"fixture-payload"})
    pr65 = _fixture_pr65_archive(tmp_path / "pr65.zip", active_pair=7)
    selected_raw = _selected_raw(tmp_path / "selected.raw")
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps({"pair_indices": [7]}, sort_keys=True) + "\n")

    summary = module.build_candidates(
        source_archive=source,
        pr65_archive=pr65,
        output_dir=tmp_path / "out",
        specs=(module.CandidateSpec("fixture_bias_pair7", ("bias",), (7,), "test"),),
        selected_raw_path=selected_raw,
        parity_json=parity,
        raw_height=2,
        raw_width=3,
        frontier_eval_json=None,
        exact_negative_globs=(),
        expected_source_sha256=None,
        expected_pr65_sha256=None,
        require_runtime_probe=False,
    )

    candidate = summary["candidates"][0]
    assert candidate["candidate_id"] == "fixture_bias_pair7"
    assert candidate["raw_changed_values"] > 0
    archive = tmp_path / "out" / "fixture_bias_pair7" / "archive.zip"
    assert archive.is_file()
    with zipfile.ZipFile(archive, "r") as zf:
        assert sorted(zf.namelist()) == ["p", "qpost.bin"]
    manifest = json.loads((tmp_path / "out" / "fixture_bias_pair7" / "manifest.json").read_text())
    assert manifest["score_claim"] is False
    assert manifest["no_op_proof"]["is_noop"] is False
    assert manifest["raw_output_delta_proof"]["exact_equal"] is False
    assert manifest["dispatch_recommendation"]["remote_dispatched"] is False


def test_pr79_pr65_builder_fails_when_raw_proof_pair_missing(tmp_path: Path) -> None:
    module = _load()
    source = tmp_path / "source.zip"
    _stored_zip(source, {"p": b"fixture-payload"})
    pr65 = _fixture_pr65_archive(tmp_path / "pr65.zip", active_pair=7)
    selected_raw = _selected_raw(tmp_path / "selected.raw")
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps({"pair_indices": [8]}, sort_keys=True) + "\n")

    with pytest.raises(module.PostprocessAtomBuildError, match="raw proof set"):
        module.build_candidates(
            source_archive=source,
            pr65_archive=pr65,
            output_dir=tmp_path / "out",
            specs=(module.CandidateSpec("fixture_bias_pair7", ("bias",), (7,), "test"),),
            selected_raw_path=selected_raw,
            parity_json=parity,
            raw_height=2,
            raw_width=3,
            frontier_eval_json=None,
            exact_negative_globs=(),
            expected_source_sha256=None,
            expected_pr65_sha256=None,
            require_runtime_probe=False,
        )


def test_pr79_pr65_builder_fails_closed_on_identity_atoms(tmp_path: Path) -> None:
    module = _load()
    source = tmp_path / "source.zip"
    _stored_zip(source, {"p": b"fixture-payload"})
    pr65 = _fixture_pr65_archive(tmp_path / "pr65.zip", active_pair=7)
    selected_raw = _selected_raw(tmp_path / "selected.raw")
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps({"pair_indices": [7]}, sort_keys=True) + "\n")

    with pytest.raises(module.PostprocessAtomBuildError, match="selected atom set is no-op"):
        module.build_candidates(
            source_archive=source,
            pr65_archive=pr65,
            output_dir=tmp_path / "out",
            specs=(module.CandidateSpec("fixture_shift_pair7", ("shift",), (7,), "test"),),
            selected_raw_path=selected_raw,
            parity_json=parity,
            raw_height=2,
            raw_width=3,
            frontier_eval_json=None,
            exact_negative_globs=(),
            expected_source_sha256=None,
            expected_pr65_sha256=None,
            require_runtime_probe=False,
        )
