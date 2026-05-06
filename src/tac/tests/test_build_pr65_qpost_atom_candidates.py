from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import brotli
import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_pr65_qpost_atom_candidates.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_pr65_qpost_atom_candidates_test", BUILDER_PATH)
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
    _stored_zip(path, {"x": header + core + b"".join(streams[name] for name in ("post", "shift", "frac", "frac2", "frac3", "bias", "region")) + streams["randmulti"]})
    return path


def _trace(path: Path, *, pair: int, combined: float) -> Path:
    samples = []
    for idx in range(600):
        value = combined if idx == pair else 0.0
        samples.append(
            {
                "pair_index": idx,
                "score_combined_contribution_first_order": value,
                "score_pose_contribution_first_order": value / 2.0,
                "score_seg_contribution_exact": value / 2.0,
                "posenet_dist": 0.0,
                "segnet_dist": 0.0,
            }
        )
    path.write_text('{"samples":' + __import__("json").dumps(samples) + "}\n")
    return path


def test_qpost_atom_matrix_builds_pair_filtered_non_noop_archive(tmp_path: Path) -> None:
    module = _load()
    source = tmp_path / "source.zip"
    _stored_zip(source, {"p": b"base"})
    pr65 = _fixture_pr65_archive(tmp_path / "pr65.zip", active_pair=7)
    c089_trace = _trace(tmp_path / "c089_trace.json", pair=7, combined=0.003)
    pr65_trace = _trace(tmp_path / "pr65_trace.json", pair=7, combined=0.001)

    matrix = module.build_matrix(
        source_archive=source,
        pr65_archive=pr65,
        output_dir=tmp_path / "out",
        anatomy_json=None,
        c089_trace=c089_trace,
        pr65_trace=pr65_trace,
        specs=(module.CandidateSpec("tiny_bias", ("bias",), 1, "test"),),
        expected_source_sha256=None,
        expected_pr65_sha256=None,
        expected_pr65_head_sha=None,
    )

    candidate = matrix["candidate_summary"][0]
    assert candidate["built"] is True
    assert candidate["candidate_id"] == "tiny_bias"
    assert candidate["selected_pairs"] == [7]
    assert candidate["selected_active_atoms_total"] == 1
    assert Path(candidate["archive"]).is_file()
    manifest = __import__("json").loads((tmp_path / "out" / "tiny_bias" / "manifest.json").read_text())
    assert manifest["no_op_proof"]["is_noop"] is False
    assert manifest["qpost_streams"]["bias"]["bytes"] > 0
    assert manifest["qpost_streams"]["randmulti"]["bytes"] == 0


def test_qpost_atom_matrix_skips_selected_noop(tmp_path: Path) -> None:
    module = _load()
    source = tmp_path / "source.zip"
    _stored_zip(source, {"p": b"base"})
    pr65 = _fixture_pr65_archive(tmp_path / "pr65.zip", active_pair=7)
    c089_trace = _trace(tmp_path / "c089_trace.json", pair=7, combined=0.0)
    pr65_trace = _trace(tmp_path / "pr65_trace.json", pair=7, combined=0.0)

    matrix = module.build_matrix(
        source_archive=source,
        pr65_archive=pr65,
        output_dir=tmp_path / "out",
        anatomy_json=None,
        c089_trace=c089_trace,
        pr65_trace=pr65_trace,
        specs=(module.CandidateSpec("tiny_shift", ("shift",), 1, "test"),),
        positive_trace_only=False,
        expected_source_sha256=None,
        expected_pr65_sha256=None,
        expected_pr65_head_sha=None,
    )

    candidate = matrix["candidate_summary"][0]
    assert candidate["built"] is False
    assert candidate["skip_reason"] == "no eligible active pairs after ranking filters"


def test_qpost_atom_matrix_fails_closed_on_pr65_sha_mismatch(tmp_path: Path) -> None:
    module = _load()
    source = tmp_path / "source.zip"
    _stored_zip(source, {"p": b"base"})
    pr65 = _fixture_pr65_archive(tmp_path / "pr65.zip", active_pair=7)
    with pytest.raises(module.QPostAtomPlannerError, match="PR65 source SHA mismatch"):
        module.build_matrix(
            source_archive=source,
            pr65_archive=pr65,
            output_dir=tmp_path / "out",
            anatomy_json=None,
            c089_trace=None,
            pr65_trace=None,
            specs=(module.CandidateSpec("tiny_bias", ("bias",), 1, "test"),),
            expected_source_sha256=None,
            expected_pr65_sha256="not-the-sha",
            expected_pr65_head_sha=None,
        )


def test_custom_spec_rejects_randmulti_pair_filter() -> None:
    module = _load()
    with pytest.raises(module.QPostAtomPlannerError, match="unsupported custom qpost stream"):
        module.parse_specs(["bad:randmulti:1"])
