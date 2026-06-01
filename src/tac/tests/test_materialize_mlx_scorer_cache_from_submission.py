from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import zipfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import torch

from tac.local_acceleration.mlx_preprocess import CAMERA_HW
from tac.substrates._shared.inflate_runtime import (
    rgb_pair_to_uint8_frames,
    write_rgb_pair_to_raw,
)
from tac.substrates.hi_nerv.architecture import HinervConfig, HinervSubstrate
from tac.substrates.hi_nerv.archive import pack_archive as pack_hi_nerv_archive
from tac.substrates.hi_nerv.inflate import build_model_from_archive
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


def test_rgb_pair_to_uint8_frames_matches_raw_writer_lowering() -> None:
    rgb_0 = torch.linspace(0.0, 1.0, 3 * 4 * 5, dtype=torch.float32).reshape(
        1, 3, 4, 5
    )
    rgb_1 = torch.flip(rgb_0, dims=(-1,))

    lowered = rgb_pair_to_uint8_frames(rgb_0, rgb_1, input_range="unit")
    buf = io.BytesIO()
    n = write_rgb_pair_to_raw(buf, rgb_0, rgb_1, input_range="unit")

    assert n == 2
    assert lowered.shape == (2, CAMERA_HW[0], CAMERA_HW[1], 3)
    assert buf.getvalue() == lowered.tobytes(order="C")


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


def test_submission_mlx_cache_can_reuse_preinflated_png_frame_tree(
    tmp_path: Path,
) -> None:
    from PIL import Image  # type: ignore[import-not-found]

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
    frame_tree = preinflated / "0.raw"
    frame_tree.mkdir(parents=True)
    for frame_idx in range(4):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :, frame_idx % 3] = 17 + frame_idx
        Image.fromarray(frame).save(frame_tree / f"{frame_idx}.png")
    proof_manifest = tmp_path / "preinflated_png_proof.json"
    proof_manifest.write_text(
        json.dumps(
            {
                "schema": "test_preinflated_png_tree_proof.v1",
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "file_list_path": str(video_names),
                "receiver_output_path": str(frame_tree),
                "runtime_consumption_proof_passed": True,
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

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
            "--preinflated-proof-manifest",
            str(proof_manifest),
            "--allow-png-frame-tree-output",
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
    inflated_manifest = json.loads(
        (work_dir / "inflated_outputs_manifest.json").read_text(encoding="utf-8")
    )

    assert stdout["cached_pair_count"] == 1
    assert payload["inflate_executed"] is False
    assert payload["allow_png_frame_tree_output"] is True
    assert payload["preinflated_output_dir"] == str(preinflated)
    assert payload["preinflated_proof_manifest"] == str(proof_manifest)
    assert payload["raw_path"] == str(frame_tree)
    assert payload["inflated_surface_kind"] == "png_frame_tree"
    assert payload["png_frame_count"] == 4
    assert payload["png_tree_cache_blockers"] == [
        "png_frame_tree_cache_prefix_subset",
        "png_frame_tree_frame_count_4_not_1200",
        "png_frame_tree_noncontest_raw_geometry_24x32",
    ]
    assert payload["raw_pair_count"] == 2
    assert payload["cached_pair_count"] == 1
    assert payload["score_claim"] is False
    assert manifest["source"] == str(frame_tree)
    assert manifest["source_kind"] == "png_frame_tree_inflate"
    assert manifest["pair_count"] == 1
    assert manifest["frame_shape_hwc"] == [24, 32, 3]
    assert manifest["png_frame_tree_contract"]["cache_blockers"] == [
        "png_frame_tree_cache_prefix_subset",
        "png_frame_tree_frame_count_4_not_1200",
        "png_frame_tree_noncontest_raw_geometry_24x32",
    ]
    assert manifest["raw_sha256_scope"] == "cached_pair_stream"
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert inflated_manifest["surface_kind"] == "png_frame_tree"
    assert inflated_manifest["png_frame_count"] == 4
    assert inflated_manifest["score_claim"] is False
    assert not (work_dir / "inflated").exists()


def test_submission_mlx_cache_rejects_png_frame_tree_without_explicit_opt_in(
    tmp_path: Path,
) -> None:
    from PIL import Image  # type: ignore[import-not-found]

    archive = tmp_path / "archive.zip"
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "inflate.sh").write_text("#!/usr/bin/env bash\nexit 99\n")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("data.bin", b"payload")
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    video_names = upstream / "public_test_video_names.txt"
    video_names.write_text("0.mkv\n", encoding="utf-8")
    frame_tree = tmp_path / "preinflated" / "0.raw"
    frame_tree.mkdir(parents=True)
    for frame_idx in range(2):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            frame_tree / f"{frame_idx}.png"
        )

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
            str(tmp_path / "cache"),
            "--work-dir",
            str(tmp_path / "work"),
            "--report-output",
            str(tmp_path / "report.json"),
            "--preinflated-output-dir",
            str(tmp_path / "preinflated"),
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
        check=False,
    )

    assert result.returncode != 0
    assert "--allow-png-frame-tree-output" in result.stderr


def test_submission_mlx_cache_rejects_preinflated_png_without_proof(
    tmp_path: Path,
) -> None:
    from PIL import Image  # type: ignore[import-not-found]

    archive = tmp_path / "archive.zip"
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "inflate.sh").write_text("#!/usr/bin/env bash\nexit 99\n")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("data.bin", b"payload")
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    video_names = upstream / "public_test_video_names.txt"
    video_names.write_text("0.mkv\n", encoding="utf-8")
    frame_tree = tmp_path / "preinflated" / "0.raw"
    frame_tree.mkdir(parents=True)
    for frame_idx in range(2):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            frame_tree / f"{frame_idx}.png"
        )

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
            str(tmp_path / "cache"),
            "--work-dir",
            str(tmp_path / "work"),
            "--report-output",
            str(tmp_path / "report.json"),
            "--preinflated-output-dir",
            str(tmp_path / "preinflated"),
            "--allow-png-frame-tree-output",
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
        check=False,
    )

    assert result.returncode != 0
    assert "--preinflated-proof-manifest is required" in result.stderr


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
    assert (
        payload["candidate_cache_identity_mode"]
        == "hprc_direct_receiver_render_cache_identity_audited_false_authority"
    )
    stamp = manifest["hprc_direct_receiver_render_cache_identity_audit"]
    audit_path = Path(stamp["path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert stamp["verdict"] == "PASS_HPRC_DIRECT_RECEIVER_RENDER_CACHE_IDENTITY"
    assert stamp["score_claim"] is False
    assert stamp["ready_for_exact_eval_dispatch"] is False
    assert audit["cache"]["raw_sha256"] == manifest["raw_sha256"]
    assert audit["cache"]["array_sha256"] == manifest["array_sha256"]
    assert audit["receiver_proof_required_for_promotion"] is True
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
    stamp = manifest["hprc_direct_receiver_render_cache_identity_audit"]
    audit = json.loads(Path(stamp["path"]).read_text(encoding="utf-8"))
    assert audit["direct_render"]["selected_pair_ranges"] == [[1, 1]]
    assert audit["direct_render"]["pair_index_scope"] == "explicit_pair_ranges"
    assert manifest["pair_count"] == 1
    assert pair_indices.tolist() == [[2, 3]]
    assert manifest["raw_sha256"] == hashlib.sha256(expected_raw).hexdigest()


def test_submission_mlx_cache_can_render_hi_nerv_direct_without_raw_scratch(
    tmp_path: Path,
) -> None:
    cfg = HinervConfig(
        latent_dim_coarse=2,
        latent_dim_mid=2,
        latent_dim_fine=2,
        embed_dim=2,
        initial_grid_h=1,
        initial_grid_w=1,
        decoder_channels=(2, 2, 2),
        sin_frequency=3.0,
        num_upsample_blocks=3,
        mid_injection_block_index=0,
        fine_injection_block_index=1,
        num_pairs=2,
        output_height=8,
        output_width=8,
    )
    torch.manual_seed(7)
    model = HinervSubstrate(cfg).eval()
    meta = {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "mid_injection_block_index": cfg.mid_injection_block_index,
        "fine_injection_block_index": cfg.fine_injection_block_index,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }
    packet_bytes = pack_hi_nerv_archive(
        dict(model.state_dict()),
        model.latents_coarse.detach(),
        model.latents_mid.detach(),
        model.latents_fine.detach(),
        meta,
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
            "--receiver-direct-cache",
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

    stdout = json.loads(result.stdout)
    payload = json.loads(report.read_text(encoding="utf-8"))
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    _, _, receiver_model = build_model_from_archive(packet_bytes, device="cpu")
    with torch.no_grad():
        rgb_0, rgb_1 = receiver_model(torch.tensor([1], dtype=torch.long))
    expected_raw = rgb_pair_to_uint8_frames(
        rgb_0,
        rgb_1,
        input_range="unit",
    ).tobytes(order="C")
    pair_indices = np.load(cache_dir / "pair_indices.npy")

    assert stdout["cached_pair_count"] == 1
    assert payload["inflate_executed"] is False
    assert payload["receiver_direct_cache"] is True
    assert payload["hprc_direct_cache"] is False
    assert payload["raw_path"] is None
    assert payload["raw_pair_count"] == 2
    assert payload["cached_pair_count"] == 1
    assert payload["direct_receiver_cache_report"]["source_family"] == "hi_nerv"
    assert payload["direct_receiver_cache_report"]["selected_pair_ranges"] == [[1, 1]]
    assert (
        payload["candidate_cache_identity_mode"]
        == "hi_nerv_direct_receiver_render_cache_identity_audited_false_authority"
    )
    stamp = manifest["hi_nerv_direct_receiver_render_cache_identity_audit"]
    audit = json.loads(Path(stamp["path"]).read_text(encoding="utf-8"))
    assert stamp["verdict"] == "PASS_HI_NERV_DIRECT_RECEIVER_RENDER_CACHE_IDENTITY"
    assert stamp["score_claim"] is False
    assert audit["source"]["archive_magic"] == "HIV1"
    assert audit["cache"]["raw_sha256"] == manifest["raw_sha256"]
    assert audit["direct_render"]["pair_index_scope"] == "explicit_pair_ranges"
    assert audit["direct_render"]["lowering"] == (
        "rgb_pair_to_uint8_frames_input_range_unit_bicubic"
    )
    assert manifest["source_kind"] == "hi_nerv_direct_receiver_render"
    assert manifest["pair_count"] == 1
    assert pair_indices.tolist() == [[2, 3]]
    assert manifest["raw_sha256"] == hashlib.sha256(expected_raw).hexdigest()
    assert not (work_dir / "inflated").exists()
