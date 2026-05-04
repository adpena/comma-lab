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
BUILDER_PATH = REPO / "experiments" / "build_cmg2_foveated_repair_candidates.py"
CMG2_BUILDER_PATH = REPO / "experiments" / "build_cmg2_downsample_candidate.py"
UNPACKER_PATH = REPO / "submissions" / "robust_current" / "unpack_renderer_payload.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)


def test_residual_atoms_rank_hard_pair_fovea_signal() -> None:
    builder = _load(BUILDER_PATH, "_cmg2_foveated_unit_builder")

    target = np.zeros((4, 8, 8), dtype=np.uint8)
    candidate = np.zeros_like(target)
    target[2, 3:5, 3:5] = 2
    target[0, 0, 0] = 3
    pair_priors = {
        "pair_signal": [1.0, 20.0],
        "hardest_pair_indices": [1],
    }

    atoms, _groups = builder.build_atom_table(
        target=target,
        candidate=candidate,
        pair_priors=pair_priors,
        repair_compressor="zlib",
        inner_radius=0.45,
        mid_radius=0.75,
    )

    assert atoms[0]["pair_index"] == 1
    assert atoms[0]["fovea_band"] == "fovea"
    assert atoms[0]["class_id"] == 2
    assert atoms[0]["score_claim"] is False


def test_row_residual_run_segments_split_by_gap_class_and_band() -> None:
    builder = _load(BUILDER_PATH, "_cmg2_foveated_row_segments_builder")
    changed = np.array([False, True, True, True, False, True, True, True], dtype=bool)
    target = np.array([0, 2, 2, 3, 0, 3, 3, 3], dtype=np.uint8)
    bands = np.array([0, 0, 1, 1, 1, 1, 2, 2], dtype=np.uint8)

    assert builder._row_residual_run_segments(  # noqa: SLF001 - regression for vectorized helper
        changed_row=changed,
        target_row=target,
        band_row=bands,
    ) == [
        (1, 2, 2, 0),
        (2, 3, 2, 1),
        (3, 4, 3, 1),
        (5, 6, 3, 1),
        (6, 8, 3, 2),
    ]


def test_builds_closed_cmg2_amr1_candidate_archive(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "_cmg2_foveated_integration_builder")
    cmg2_builder = _load(CMG2_BUILDER_PATH, "_cmg2_foveated_integration_cmg2_builder")
    unpacker = _load(UNPACKER_PATH, "_cmg2_foveated_integration_unpacker")

    full = np.zeros((4, 8, 8), dtype=np.uint8)
    full[2, 3:5, 3:5] = 2
    full[0, 0, 0] = 3
    low, recon, disagreement = cmg2_builder.downsample_block_mode(full, scale_y=2, scale_x=2)
    cmg2_payload, cmg2_header = cmg2_builder.encode_cmg2_payload(
        low,
        scale_y=2,
        scale_x=2,
        compressor="bz2",
    )
    decoded_path = tmp_path / "decoded.npy"
    np.save(decoded_path, full)
    source_archive = tmp_path / "cmg2_source_members.zip"
    _write_zip(
        source_archive,
        [
            ("renderer.bin", b"QZS3fake"),
            ("masks.cmg2", cmg2_payload),
            ("optimized_poses.bin", struct.pack("<" + "e" * 12, *([20.0, 0.0, 0.0, 0.0, 0.0, 0.0] * 2))),
        ],
    )
    manifest = {
        "schema": "cmg2_downsample_candidate_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "canonical_score_source_required": (
            "archive.zip -> inflate.sh -> upstream/evaluate.py via "
            "experiments/contest_auth_eval.py --device cuda"
        ),
        "decoded_mask_array": {
            "path": str(decoded_path),
            "tensor_sha256": _sha256(full.tobytes(order="C")),
        },
        "cmg2": {
            **cmg2_header,
            "scale": [2, 2],
            "reconstructed_tensor_sha256": _sha256(recon.tobytes(order="C")),
            "pixel_disagreement_vs_frontier_masks": disagreement,
        },
        "source_archive": {
            "path": str(source_archive),
            "bytes": source_archive.stat().st_size,
            "sha256": _sha256(source_archive.read_bytes()),
        },
    }
    manifest_path = tmp_path / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    plan = builder.build_candidates(
        cmg2_manifest_path=manifest_path,
        output_dir=tmp_path / "out",
        component_trace=None,
        repair_compressor="zlib",
        top_policy_counts=(1, 2),
        max_atoms=8,
        plan_only=False,
        force=False,
        inner_radius=0.45,
        mid_radius=0.75,
    )

    assert plan["score_claim"] is False
    assert len(plan["candidate_manifests"]) == 2
    first_archive = Path(plan["candidate_manifests"][0]["archive"]["path"])
    with zipfile.ZipFile(first_archive) as zf:
        assert zf.namelist() == ["p"]
        payload = zf.read("p")
    _header, members = unpacker._parse_payload(__import__("brotli").decompress(payload))
    assert set(members) == {
        "renderer.bin",
        "masks.cmg2",
        "optimized_poses.bin",
        "alpha4_residual_repair.amr1.zlib",
    }
    candidate_manifest = json.loads(
        (first_archive.parent / "build_manifest.json").read_text()
    )
    assert candidate_manifest["score_claim"] is False
    assert candidate_manifest["repair"]["selection"]["partial_repair"] is True
    assert candidate_manifest["agreement"]["residual_pixels_after"] < candidate_manifest["agreement"][
        "residual_pixels_before"
    ]
