# SPDX-License-Identifier: MIT
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
BUILDER_PATH = REPO / "experiments" / "build_pr65_qpost_atom_candidates_v2.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_pr65_qpost_atom_candidates_v2_test", BUILDER_PATH)
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


def _fixture_pr65_archive(path: Path, *, active_pairs: tuple[int, ...] = (7, 9)) -> Path:
    post = np.zeros((4, 600), dtype=np.uint8)
    bias = np.full(600, 13, dtype=np.uint8)
    region = np.zeros(600, dtype=np.uint8)
    for offset, pair in enumerate(active_pairs):
        post[offset % 4, pair] = 2
        bias[pair] = 14 + offset
        region[pair] = 3
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


def _trace(path: Path, opportunities: dict[int, float], *, baseline_extra: float = 0.0) -> Path:
    samples = []
    for idx in range(600):
        value = opportunities.get(idx, 0.0) + baseline_extra
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
    path.write_text(json.dumps({"samples": samples}, sort_keys=True) + "\n")
    return path


def test_v2_builds_expansion_and_writes_overlay_manifest(tmp_path: Path) -> None:
    module = _load()
    source = tmp_path / "source.zip"
    _stored_zip(source, {"p": b"base"})
    pr65 = _fixture_pr65_archive(tmp_path / "pr65.zip", active_pairs=(7, 9))
    c089_trace = _trace(tmp_path / "c089_trace.json", {7: 0.0030, 9: 0.0024})
    pr65_trace = _trace(tmp_path / "pr65_trace.json", {7: 0.0010, 9: 0.0009})

    summary = module.build_expansion(
        source_archive=source,
        pr65_archive=pr65,
        output_dir=tmp_path / "out",
        anatomy_json=None,
        c089_trace=c089_trace,
        pr65_trace=pr65_trace,
        families=(
            module.ExpansionFamily(
                "bias_poseadv",
                ("bias",),
                (1, 2),
                "test_low",
                "low_bias_only",
            ),
        ),
        min_primary_slack=0.0002,
        allow_source_sha_mismatch=True,
    )

    assert summary["score_claim"] is False
    assert summary["remote_dispatch"]["dispatched"] is False
    assert summary["dispatch_summary"]["primary_candidate_id"] == "v2_pr65_qpost_bias_poseadv_top001"
    candidate = summary["candidate_screens"][0]
    archive = Path(candidate["archive"])
    assert archive.is_file()
    overlay = json.loads((archive.parent / "v2_manifest.json").read_text())
    assert overlay["score_claim"] is False
    assert overlay["remote_dispatch"]["requires_lane_claim"] is True
    assert str(tmp_path / "out" / "exact_eval_work") in overlay["exact_eval_command_template"]


def test_v2_keeps_post_stream_candidates_diagnostic_only(tmp_path: Path) -> None:
    module = _load()
    source = tmp_path / "source.zip"
    _stored_zip(source, {"p": b"base"})
    pr65 = _fixture_pr65_archive(tmp_path / "pr65.zip", active_pairs=(7,))
    c089_trace = _trace(tmp_path / "c089_trace.json", {7: 0.004})
    pr65_trace = _trace(tmp_path / "pr65_trace.json", {7: 0.0})

    summary = module.build_expansion(
        source_archive=source,
        pr65_archive=pr65,
        output_dir=tmp_path / "out",
        anatomy_json=None,
        c089_trace=c089_trace,
        pr65_trace=pr65_trace,
        families=(
            module.ExpansionFamily(
                "post_bias_poseadv",
                ("post", "bias"),
                (1,),
                "test_high",
                "diagnostic_post",
            ),
        ),
        allow_source_sha_mismatch=True,
    )

    screen = summary["candidate_screens"][0]
    assert screen["recommendation_class"] == "do_not_dispatch"
    assert any("contains post stream" in blocker for blocker in screen["core_dispatch_blockers"])
    assert summary["dispatch_summary"]["primary_candidate_id"] is None


def test_v2_rejects_duplicate_candidate_ids() -> None:
    module = _load()
    core = module._load_core_builder()
    family = module.ExpansionFamily("bias_poseadv", ("bias",), (1,), "risk", "low_bias_only")
    with pytest.raises(module.QPostAtomV2Error, match="duplicate candidate id"):
        module.expand_specs(core, (family, family))
