from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

from tac.local_acceleration.mlx_preprocess import CAMERA_HW
from tac.substrates.hprc.archive import parse_hprc_packet
from tac.substrates.hprc.learned_receiver import (
    build_compact_receiver_packet_from_lowres_frames,
    decode_compact_receiver_packet,
    render_compact_receiver_frame_batch,
)

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "materialize_mlx_scorer_cache_from_submission.py"


def _load_tool_module():
    for path in (REPO, REPO / "tools"):
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)
    spec = spec_from_file_location("materialize_mlx_scorer_cache_from_submission", TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_acquisition_partial_raw_uses_full_video_pair_floor() -> None:
    module = _load_tool_module()
    assert (
        module._local_acquisition_is_partial_raw(  # pyright: ignore[reportPrivateUsage]
            raw_pair_count=599,
            local_acquisition_max_pairs=600,
            inflate_executed=True,
        )
        is True
    )
    assert (
        module._local_acquisition_is_partial_raw(  # pyright: ignore[reportPrivateUsage]
            raw_pair_count=600,
            local_acquisition_max_pairs=600,
            inflate_executed=True,
        )
        is False
    )


def test_submission_mlx_cache_can_reuse_preinflated_receiver_output(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "inflate.sh").write_text(
        "#!/usr/bin/env bash\n"
        "echo should-not-run >&2\n"
        "exit 99\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("data.bin", b"payload")

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    video_names = upstream / "public_test_video_names.txt"
    video_names.write_text("0.mkv\n", encoding="utf-8")
    preinflated = tmp_path / "preinflated"
    preinflated.mkdir()
    h, w = CAMERA_HW
    raw = np.zeros((2, h, w, 3), dtype=np.uint8)
    raw[1, :, :, 0] = 7
    (preinflated / "0.raw").write_bytes(raw.tobytes())

    report = tmp_path / "report.json"
    cache_dir = tmp_path / "cache"
    work_dir = tmp_path / "work"
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--archive",
            str(archive),
            "--submission-dir",
            str(submission),
            "--upstream-dir",
            str(upstream),
            "--video-names-file",
            str(video_names),
            "--output-cache-dir",
            str(cache_dir),
            "--work-dir",
            str(work_dir),
            "--report-output",
            str(report),
            "--preinflated-output-dir",
            str(preinflated),
            "--max-pairs",
            "1",
            "--batch-pairs",
            "1",
            "--allow-large-tensor-cache",
            "--force",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )

    stdout = json.loads(result.stdout)
    payload = json.loads(report.read_text(encoding="utf-8"))
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    assert stdout["cached_pair_count"] == 1
    assert payload["inflate_executed"] is False
    assert payload["preinflated_output_dir"] == str(preinflated)
    assert payload["raw_path"] == str(preinflated / "0.raw")
    assert payload["raw_pair_count"] == 1
    assert payload["cached_pair_count"] == 1
    assert payload["local_acquisition_partial_raw"] is False
    assert payload["score_claim"] is False
    assert manifest["pair_count"] == 1
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert not (work_dir / "inflated").exists()


def test_submission_mlx_cache_can_render_hprc_direct_without_raw_scratch(
    tmp_path: Path,
) -> None:
    frames = np.zeros((4, 8, 10, 3), dtype=np.float32)
    frames[:, :, :, 0] = np.arange(4, dtype=np.float32)[:, None, None] * 11
    packet_bytes = build_compact_receiver_packet_from_lowres_frames(
        frames,
        basis_count=2,
        residual_grid_h=2,
        residual_grid_w=3,
    )
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", packet_bytes)

    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "inflate.sh").write_text(
        "#!/usr/bin/env bash\nexit 99\n",
        encoding="utf-8",
    )
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    video_names = upstream / "public_test_video_names.txt"
    video_names.write_text("0.mkv\n", encoding="utf-8")
    report = tmp_path / "report.json"
    cache_dir = tmp_path / "cache"
    work_dir = tmp_path / "work"

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--archive",
            str(archive),
            "--submission-dir",
            str(submission),
            "--upstream-dir",
            str(upstream),
            "--video-names-file",
            str(video_names),
            "--output-cache-dir",
            str(cache_dir),
            "--work-dir",
            str(work_dir),
            "--report-output",
            str(report),
            "--hprc-direct-cache",
            "--max-pairs",
            "1",
            "--batch-pairs",
            "1",
            "--allow-large-tensor-cache",
            "--force",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )

    stdout = json.loads(result.stdout)
    payload = json.loads(report.read_text(encoding="utf-8"))
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    compact = decode_compact_receiver_packet(parse_hprc_packet(packet_bytes))
    h, w = CAMERA_HW
    expected_raw = render_compact_receiver_frame_batch(
        compact,
        0,
        2,
        height=h,
        width=w,
    ).tobytes()

    assert stdout["cached_pair_count"] == 1
    assert payload["inflate_executed"] is False
    assert payload["hprc_direct_cache"] is True
    assert payload["raw_path"] is None
    assert payload["raw_pair_count"] == 2
    assert payload["cached_pair_count"] == 1
    assert payload["score_claim"] is False
    assert payload["hprc_direct_cache_report"]["receiver_proof_required_for_promotion"] is True
    assert manifest["source_kind"] == "hprc_direct_receiver_render"
    assert manifest["pair_count"] == 1
    assert manifest["raw_sha256"] == hashlib.sha256(expected_raw).hexdigest()
    assert not (work_dir / "inflated").exists()


def test_submission_mlx_cache_hprc_direct_can_render_pair_subset(
    tmp_path: Path,
) -> None:
    frames = np.zeros((4, 8, 10, 3), dtype=np.float32)
    frames[:, :, :, 0] = np.arange(4, dtype=np.float32)[:, None, None] * 13
    packet_bytes = build_compact_receiver_packet_from_lowres_frames(
        frames,
        basis_count=2,
        residual_grid_h=2,
        residual_grid_w=3,
    )
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", packet_bytes)

    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "inflate.sh").write_text("#!/usr/bin/env bash\nexit 99\n")
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    video_names = upstream / "public_test_video_names.txt"
    video_names.write_text("0.mkv\n", encoding="utf-8")
    report = tmp_path / "report.json"
    cache_dir = tmp_path / "cache"

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--archive",
            str(archive),
            "--submission-dir",
            str(submission),
            "--upstream-dir",
            str(upstream),
            "--video-names-file",
            str(video_names),
            "--output-cache-dir",
            str(cache_dir),
            "--work-dir",
            str(tmp_path / "work"),
            "--report-output",
            str(report),
            "--hprc-direct-cache",
            "--pair-ranges",
            "1",
            "--batch-pairs",
            "1",
            "--allow-large-tensor-cache",
            "--force",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    compact = decode_compact_receiver_packet(parse_hprc_packet(packet_bytes))
    h, w = CAMERA_HW
    expected_raw = render_compact_receiver_frame_batch(
        compact,
        2,
        2,
        height=h,
        width=w,
    ).tobytes()
    pair_indices = np.load(cache_dir / "pair_indices.npy")

    assert payload["cached_pair_count"] == 1
    assert payload["hprc_direct_cache_report"]["selected_pair_ranges"] == [[1, 1]]
    assert payload["hprc_direct_cache_report"]["pair_index_scope"] == "explicit_pair_ranges"
    assert manifest["pair_count"] == 1
    assert pair_indices.tolist() == [[2, 3]]
    assert manifest["raw_sha256"] == hashlib.sha256(expected_raw).hexdigest()
