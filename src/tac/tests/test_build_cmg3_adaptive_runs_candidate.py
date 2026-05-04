from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_cmg3_adaptive_runs_candidate.py"
PACKER_PATH = REPO / "experiments" / "build_renderer_packed_payload_archive.py"
INFLATE_PATH = REPO / "submissions" / "robust_current" / "inflate_renderer.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _toy_masks() -> np.ndarray:
    masks = np.zeros((1, 384, 512), dtype=np.uint8)
    masks[0, 10, 0:120] = 1
    masks[0, 10, 200:280] = 2
    masks[0, 20, 0:120] = 1
    masks[0, 20, 300:305] = 4
    return masks


def test_cmg3a_ranked_nonzero_runs_matches_base_helper() -> None:
    builder = _load(BUILDER_PATH, "_cmg3a_builder_ranked_runs_test")
    rows = [
        np.zeros(builder.WIDTH, dtype=np.uint8),
        np.full(builder.WIDTH, 3, dtype=np.uint8),
        np.array([0, 1, 1, 0, 2, 2, 2, 4] + [0] * (builder.WIDTH - 8), dtype=np.uint8),
        np.tile(np.array([0, 1, 2, 0, 4, 4, 3, 0], dtype=np.uint8), builder.WIDTH // 8),
    ]

    for row in rows:
        expected = sorted(
            builder.BASE._row_nonzero_runs(row),  # noqa: SLF001 - equivalence guard for vectorized hotpath
            key=lambda item: (-int(item[3]), int(item[0]), int(item[1]), int(item[2])),
        )
        assert builder._ranked_nonzero_runs(row) == expected


def test_cmg3a_adaptive_selector_is_deterministic() -> None:
    builder = _load(BUILDER_PATH, "_cmg3a_builder_determinism_test")
    masks = _toy_masks()

    kwargs = {
        "target_extra_runs": 2,
        "adaptive_max_runs_per_row": 3,
        "compressor": "raw",
        "hard_frame_indices": {0},
        "class_weights": {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        "foveal_row_weight": 0.0,
        "foveal_col_weight": 0.0,
        "boundary_detail_weight": 0.0,
        "rank_decay": 1.0,
    }
    stream_a, recon_a, stats_a, policy_a = builder.encode_adaptive_run_stream(masks, **kwargs)
    stream_b, recon_b, stats_b, policy_b = builder.encode_adaptive_run_stream(masks, **kwargs)

    assert stream_a == stream_b
    np.testing.assert_array_equal(recon_a, recon_b)
    assert stats_a == stats_b
    assert policy_a == policy_b


def test_cmg3a_adaptive_adds_one_valuable_second_run_without_global_top2() -> None:
    builder = _load(BUILDER_PATH, "_cmg3a_builder_selection_test")
    masks = _toy_masks()

    _stream, recon, stats, policy = builder.encode_adaptive_run_stream(
        masks,
        target_extra_runs=1,
        adaptive_max_runs_per_row=3,
        compressor="raw",
        class_weights={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        foveal_row_weight=0.0,
        foveal_col_weight=0.0,
        boundary_detail_weight=0.0,
        rank_decay=1.0,
    )

    assert stats["base_top1_runs"] == 2
    assert stats["selected_extra_runs"] == 1
    assert stats["rows_with_extra_runs"] == 1
    assert stats["selected_rows_by_run_count"]["2"] == 1
    assert stats["selected_rows_by_run_count"]["1"] == 1
    assert stats["selected_extra_rank_histogram"] == {"2": 1}
    assert policy["row_rank_prefix_required"] is False
    np.testing.assert_array_equal(recon[0, 10, 200:280], np.full(80, 2, dtype=np.uint8))
    np.testing.assert_array_equal(recon[0, 20, 300:305], np.zeros(5, dtype=np.uint8))


def test_cmg3a_can_consume_field_equation_policy_kwarg() -> None:
    builder = _load(BUILDER_PATH, "_cmg3a_builder_field_policy_test")
    masks = _toy_masks()
    field_policy = {
        "source_path": "/tmp/field_policy.json",
        "source_sha256": "a" * 64,
        "source_schema": "yousfi_fridrich_atom_field_allocator_v1",
        "source_mode": "contest",
        "policy_id": "unit_policy",
        "policy": {
            "policy_id": "unit_policy",
            "required_base_runs_per_row": 1,
            "selected_row_run_atoms": [
                {
                    "frame_index": 0,
                    "y": 20,
                    "x0": 300,
                    "x1_exclusive": 305,
                    "class_id": 4,
                },
                {
                    "frame_index": 0,
                    "y": 10,
                    "x0": 0,
                    "x1_exclusive": 120,
                    "class_id": 1,
                },
            ],
        },
    }

    _stream, recon, stats, policy = builder.encode_adaptive_run_stream(
        masks,
        target_extra_runs=None,
        adaptive_max_runs_per_row=3,
        compressor="raw",
        class_weights={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        foveal_row_weight=0.0,
        foveal_col_weight=0.0,
        boundary_detail_weight=0.0,
        rank_decay=1.0,
        field_atom_policy=field_policy,
    )

    np.testing.assert_array_equal(recon[0, 20, 300:305], np.full(5, 4, dtype=np.uint8))
    assert stats["selected_extra_runs"] == 1
    assert policy["selector"] == "field_equation_explicit_row_run_policy_after_base_runs_per_row"
    assert policy["field_atom_policy"]["policy_id"] == "unit_policy"
    assert policy["field_atom_policy"]["matched_extra_atom_count"] == 1
    assert policy["field_atom_policy"]["already_base_atom_count"] == 1
    assert policy["field_atom_policy"]["source_basename"] == "field_policy.json"
    assert "source_path" not in policy["field_atom_policy"]


def test_cmg3a_field_policy_fails_closed_on_base_mismatch_and_unmatched_atoms() -> None:
    builder = _load(BUILDER_PATH, "_cmg3a_builder_field_policy_guard_test")
    masks = _toy_masks()
    base_policy = {
        "policy_id": "unit_policy",
        "policy": {
            "policy_id": "unit_policy",
            "required_base_runs_per_row": 2,
            "selected_row_run_atoms": [
                {
                    "frame_index": 0,
                    "y": 20,
                    "x0": 300,
                    "x1_exclusive": 305,
                    "class_id": 4,
                }
            ],
        },
    }

    try:
        builder.encode_adaptive_run_stream(
            masks,
            base_runs_per_row=1,
            adaptive_max_runs_per_row=3,
            compressor="raw",
            class_weights={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            foveal_row_weight=0.0,
            foveal_col_weight=0.0,
            boundary_detail_weight=0.0,
            rank_decay=1.0,
            field_atom_policy=base_policy,
        )
    except ValueError as exc:
        assert "requires base_runs_per_row=2" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected base mismatch failure")

    unmatched_policy = {
        "policy_id": "unit_policy",
        "policy": {
            "policy_id": "unit_policy",
            "required_base_runs_per_row": 1,
            "selected_row_run_atoms": [
                {
                    "frame_index": 0,
                    "y": 20,
                    "x0": 301,
                    "x1_exclusive": 305,
                    "class_id": 4,
                }
            ],
        },
    }

    try:
        builder.encode_adaptive_run_stream(
            masks,
            base_runs_per_row=1,
            adaptive_max_runs_per_row=3,
            compressor="raw",
            class_weights={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            foveal_row_weight=0.0,
            foveal_col_weight=0.0,
            boundary_detail_weight=0.0,
            rank_decay=1.0,
            field_atom_policy=unmatched_policy,
        )
    except ValueError as exc:
        assert "do not match the selected base/candidate pool" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected unmatched atom failure")

    duplicate_policy = {
        "policy_id": "unit_policy",
        "policy": {
            "policy_id": "unit_policy",
            "required_base_runs_per_row": 1,
            "selected_row_run_atoms": [
                {
                    "frame_index": 0,
                    "y": 20,
                    "x0": 300,
                    "x1_exclusive": 305,
                    "class_id": 4,
                },
                {
                    "frame_index": 0,
                    "y": 20,
                    "x0": 300,
                    "x1_exclusive": 305,
                    "class_id": 4,
                },
            ],
        },
    }

    try:
        builder.encode_adaptive_run_stream(
            masks,
            base_runs_per_row=1,
            adaptive_max_runs_per_row=3,
            compressor="raw",
            class_weights={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            foveal_row_weight=0.0,
            foveal_col_weight=0.0,
            boundary_detail_weight=0.0,
            rank_decay=1.0,
            field_atom_policy=duplicate_policy,
        )
    except ValueError as exc:
        assert "duplicate row-run atom selections" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected duplicate atom failure")


def test_cmg3a_target_body_search_handles_nonmonotonic_compressed_sizes() -> None:
    builder = _load(BUILDER_PATH, "_cmg3a_builder_nonmonotonic_body_search_test")
    masks = np.zeros((1, builder.HEIGHT, builder.WIDTH), dtype=np.uint8)

    base = builder.RunAtom(
        frame_index=0,
        y=0,
        flat_row=0,
        rank=1,
        start=0,
        end=9,
        class_id=1,
        length=10,
        priority=10.0,
    )
    candidates = [
        builder.RunAtom(
            frame_index=0,
            y=i,
            flat_row=i,
            rank=2,
            start=10,
            end=19,
            class_id=2,
            length=10,
            priority=float(10 - i),
        )
        for i in range(1, 6)
    ]

    body_by_extra_count = {
        0: 10,
        1: 14,
        2: 30,
        3: 12,
        4: 40,
        5: 50,
    }
    original = builder._compressed_body_bytes

    def fake_compressed_body_bytes(run_stream: bytes, _compressor: str) -> int:
        pos = 0
        total_runs = 0
        for _row in range(builder.HEIGHT):
            row_run_count = int(run_stream[pos])
            total_runs += row_run_count
            pos += 1 + 5 * row_run_count
        extra_count = max(0, total_runs - 1)
        return body_by_extra_count[extra_count]

    builder._compressed_body_bytes = fake_compressed_body_bytes
    try:
        selected, report = builder._select_extra_prefix_count(
            masks=masks,
            base_by_row={0: [base]},
            candidates=candidates,
            effective_extra_run_cap=5,
            target_body_bytes=16,
            compressor="fake",
            body_search_mode="exhaustive",
        )
    finally:
        builder._compressed_body_bytes = original

    assert selected == 3
    assert report["target_met"] is True
    assert report["selected_body_bytes_during_search"] == 12
    assert report["exhaustive_prefix_search"] is True
    assert report["monotonic_binary_search"] is False
    assert report["selected_by"] == "largest_exhaustive_priority_prefix_under_target_body_bytes"


def test_cmg3a_payload_decodes_through_existing_runtime_loader(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "_cmg3a_builder_runtime_test")
    inflate = _load(INFLATE_PATH, "_cmg3a_inflate_runtime_test")
    masks = _toy_masks()

    stream, recon, stats, policy = builder.encode_adaptive_run_stream(
        masks,
        target_extra_runs=1,
        adaptive_max_runs_per_row=3,
        compressor="raw",
        class_weights={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        foveal_row_weight=0.0,
        foveal_col_weight=0.0,
        boundary_detail_weight=0.0,
        rank_decay=1.0,
    )
    payload, header = builder.encode_cmg3a_payload(
        stream,
        frame_count=1,
        max_runs_per_row=stats["max_selected_runs_per_row"],
        source_mask_sha256=hashlib.sha256(masks.tobytes()).hexdigest(),
        reconstructed_mask_sha256=hashlib.sha256(recon.tobytes()).hexdigest(),
        pixel_disagreement=stats["pixel_disagreement"],
        pixel_disagreement_count=stats["pixel_disagreement_count"],
        policy=policy,
        compressor="raw",
    )
    path = tmp_path / "masks.cmg3"
    path.write_bytes(payload)

    decoded = inflate._load_masks_from_cmg3(path, expected_frames=1)

    np.testing.assert_array_equal(decoded.numpy(), recon)
    assert header["schema"] == "cmg3a_adaptive_nonzero_row_runs_candidate_v1"
    assert header["mode"] == "nonzero_row_runs_topk_v1"
    assert header["max_runs_per_row"] == 2
    assert header["selection_policy"]["selected_extra_runs"] == 1


def test_cmg3a_build_manifest_records_policy_and_hashes(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "_cmg3a_builder_manifest_test")
    packer = _load(PACKER_PATH, "_cmg3a_packer_manifest_test")
    masks = _toy_masks()
    mask_path = tmp_path / "decoded_masks.npy"
    np.save(mask_path, masks)

    source = tmp_path / "frontier_source.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("renderer.bin", b"QZS3fake")
        zf.writestr("masks.mkv", b"frontier-mask-bytes")
        zf.writestr("optimized_poses.bin", struct.pack("<" + "e" * 12, *([20.0, 0.0, 0.0, 0.0, 0.0, 0.0] * 2)))
    frontier_archive = tmp_path / "frontier_archive.zip"
    packer.build_packed_archive(
        source,
        frontier_archive,
        brotli_quality=1,
        payload_member_name=packer.SHORT_PAYLOAD_MEMBER_NAME,
        payload_format=packer.PAYLOAD_FORMAT_RPK1_JSON,
    )

    manifest = builder.build_candidate(
        frontier_archive=frontier_archive,
        decoded_mask_array=mask_path,
        output_dir=tmp_path / "candidate",
        target_extra_runs=1,
        adaptive_max_runs_per_row=3,
        compressor="raw",
        class_weights={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        foveal_row_weight=0.0,
        foveal_col_weight=0.0,
        boundary_detail_weight=0.0,
        rank_decay=1.0,
    )
    saved = json.loads((tmp_path / "candidate" / "build_manifest.json").read_text())

    assert saved == manifest
    assert manifest["score_claim"] is False
    assert manifest["policy"]["schema"] == "cmg3a_adaptive_global_nonzero_row_run_policy_v1"
    assert manifest["policy"]["selected_extra_runs"] == 1
    assert manifest["cmg3"]["run_stats"]["selected_extra_runs"] == 1
    assert manifest["cmg3"]["reconstructed_mask_u8_sha256"] == hashlib.sha256(
        builder.encode_adaptive_run_stream(
            masks,
            target_extra_runs=1,
            adaptive_max_runs_per_row=3,
            compressor="raw",
            class_weights={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            foveal_row_weight=0.0,
            foveal_col_weight=0.0,
            boundary_detail_weight=0.0,
            rank_decay=1.0,
        )[1].tobytes()
    ).hexdigest()
    assert len(manifest["output_archive"]["sha256"]) == 64
    assert manifest["output_archive"]["bytes"] == (tmp_path / "candidate" / "archive.zip").stat().st_size


def test_cmg3a_build_accepts_decoded_mask_npz(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "_cmg3a_builder_manifest_npz_test")
    packer = _load(PACKER_PATH, "_cmg3a_packer_manifest_npz_test")
    masks = _toy_masks()
    mask_path = tmp_path / "decoded_masks.npz"
    np.savez_compressed(mask_path, masks=masks)

    source = tmp_path / "frontier_source.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("renderer.bin", b"QZS3fake")
        zf.writestr("masks.mkv", b"frontier-mask-bytes")
        zf.writestr(
            "optimized_poses.bin",
            struct.pack("<" + "e" * 12, *([20.0, 0.0, 0.0, 0.0, 0.0, 0.0] * 2)),
        )
    frontier_archive = tmp_path / "frontier_archive.zip"
    packer.build_packed_archive(
        source,
        frontier_archive,
        brotli_quality=1,
        payload_member_name=packer.SHORT_PAYLOAD_MEMBER_NAME,
        payload_format=packer.PAYLOAD_FORMAT_RPK1_JSON,
    )

    manifest = builder.build_candidate(
        frontier_archive=frontier_archive,
        decoded_mask_array=mask_path,
        output_dir=tmp_path / "candidate",
        target_extra_runs=1,
        adaptive_max_runs_per_row=3,
        compressor="raw",
        class_weights={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        foveal_row_weight=0.0,
        foveal_col_weight=0.0,
        boundary_detail_weight=0.0,
        rank_decay=1.0,
    )

    assert manifest["decoded_mask_array"]["path"].endswith("decoded_masks.npz")
    assert manifest["cmg3"]["source_mask_u8_sha256"] == hashlib.sha256(masks.tobytes()).hexdigest()
