# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "probe_cmg2_mask_codecs.py"
SPEC = importlib.util.spec_from_file_location("probe_cmg2_mask_codecs", MODULE_PATH)
assert SPEC is not None
probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def _tiny_masks() -> np.ndarray:
    arr = np.zeros((4, 8, 12), dtype=np.uint8)
    arr[:, 4:, :] = 2
    arr[1::2, 2:4, 3:8] = 3
    arr[2, 1:5, 8:10] = 4
    return arr


def test_symbol_and_bitplane_packers_are_deterministic() -> None:
    arr = _tiny_masks()

    assert probe._pack_symbol_bits(arr) == probe._pack_symbol_bits(arr)
    assert probe._pack_bitplanes(arr) == probe._pack_bitplanes(arr)
    assert len(probe._pack_symbol_bits(arr)) == (arr.size * 3 + 7) // 8
    assert len(probe._pack_bitplanes(arr)) == 3 * ((arr.size + 7) // 8)


def test_temporal_transforms_are_lossless_invertible_shapes() -> None:
    arr = _tiny_masks()

    frame_xor = probe._frame_xor(arr)
    restored = np.empty_like(arr)
    restored[0] = frame_xor[0]
    for index in range(1, arr.shape[0]):
        restored[index] = np.bitwise_xor(frame_xor[index], restored[index - 1])
    np.testing.assert_array_equal(restored, arr)

    pair_xor = probe._pair_xor(arr)
    restored_pair = pair_xor.copy()
    restored_pair[1::2] = np.bitwise_xor(pair_xor[1::2], pair_xor[0::2])
    np.testing.assert_array_equal(restored_pair, arr)


def test_probe_report_is_non_promotable_and_baseline_aware(tmp_path: Path) -> None:
    masks_path = tmp_path / "masks.npy"
    baseline_path = tmp_path / "masks.mkv"
    np.save(masks_path, _tiny_masks())
    baseline_path.write_bytes(b"baseline-mask-stream")

    report = probe.run_probe(
        probe.ProbeConfig(
            decoded_mask_array=masks_path,
            baseline_mask_stream=baseline_path,
            output_dir=tmp_path / "out",
            compressors=("zlib9",),
            transforms=("raw_u8", "symbol_packed3", "frame_xor_symbol_packed3"),
            write_best_payload=True,
        ),
        command=["unit-test"],
    )

    manifest_path = tmp_path / "out" / probe.REPORT_NAME
    payload_path = tmp_path / "out" / probe.BEST_PAYLOAD_NAME
    assert manifest_path.exists()
    assert payload_path.exists()
    assert json.loads(manifest_path.read_text()) == report
    assert report["schema"] == "cmg2_mask_codec_probe_v1"
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["evidence_grade"] == "empirical_byte_probe_only"
    assert report["scorer_network_loaded"] is False
    assert "contest_auth_eval.py --device cuda" in report["canonical_score_source_required"]
    assert report["baseline"]["bytes"] == baseline_path.stat().st_size
    assert report["best_variant"]["baseline_delta_bytes"] == (
        report["best_variant"]["compressed_size_bytes"] - baseline_path.stat().st_size
    )
    assert report["artifacts"]["best_payload"]["runtime_ready"] is False


def test_probe_rejects_output_overwrite_without_force(tmp_path: Path) -> None:
    masks_path = tmp_path / "masks.npy"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep")
    np.save(masks_path, _tiny_masks())

    with pytest.raises(FileExistsError, match="--force"):
        probe.run_probe(
            probe.ProbeConfig(
                decoded_mask_array=masks_path,
                output_dir=output_dir,
                compressors=("zlib9",),
                transforms=("raw_u8",),
            ),
            command=["unit-test"],
        )


def test_probe_rejects_non_uint8_arrays(tmp_path: Path) -> None:
    masks_path = tmp_path / "bad.npy"
    np.save(masks_path, np.zeros((1, 2, 3), dtype=np.int16))

    with pytest.raises(ValueError, match="uint8"):
        probe.run_probe(
            probe.ProbeConfig(
                decoded_mask_array=masks_path,
                output_dir=tmp_path / "out",
                compressors=("zlib9",),
                transforms=("raw_u8",),
            ),
            command=["unit-test"],
        )
