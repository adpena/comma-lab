# SPDX-License-Identifier: MIT
"""Focused tests for the bounded V10 free-predictor floor measurement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import tools.measure_v10_free_predictor_floor as measure
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator


def _synthetic_planes() -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    y0 = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)
    y1 = y0.copy()
    y1[0, 1:3, 2:4] = np.clip(y1[0, 1:3, 2:4].astype(np.int16) + 7, 0, 255)
    y1[1] = np.roll(y1[1], 1, axis=1)
    labels = [np.arange(20, dtype=np.uint8).reshape(4, 5) % 5 for _ in range(2)]
    margins = [
        np.array(
            [0.0, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 0.07, 0.24, 0.49, 0.99, 1.99, 9.0, 0.0, 2.0],
            dtype=np.float32,
        ).reshape(4, 5)
        for _ in range(2)
    ]
    return y0, y1, labels, margins


def test_exact_operator_round_u8_uses_integer_half_up() -> None:
    operator = DisjointResizeOperator.build(camera_h=4, camera_w=4, scorer_h=2, scorer_w=2)
    frame = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    numerators, denominator = operator.apply_numerators(frame)
    expected = ((numerators.astype(np.int64) + denominator // 2) // denominator).astype(np.uint8)
    actual = measure.exact_operator_round_u8(operator, frame)
    assert actual.dtype == np.uint8
    assert actual.flags.c_contiguous
    assert np.array_equal(actual, expected)


def test_attribution_stream_counts_mask_and_values_and_refuses_trailing_data() -> None:
    residual0 = np.arange(4 * 5 * 3, dtype=np.int16).reshape(4, 5, 3).astype("<i2")
    residual1 = (-residual0).astype("<i2")
    masks = [np.indices((4, 5)).sum(axis=0) % 2 == 0, np.eye(4, 5, dtype=bool)]
    payload = measure.pack_attribution_stream(
        kind="class", bucket_id=3, pair_ids=(10, 11), masks=masks, residuals=(residual0, residual1)
    )
    parsed = measure.parse_attribution_stream(payload)
    assert parsed["kind"] == "class"
    assert parsed["bucket_id"] == 3
    assert parsed["pair_ids"] == [10, 11]
    assert parsed["selected_counts"] == [int(masks[0].sum()), int(masks[1].sum())]
    with pytest.raises(measure.PredictorFloorError, match="trailing"):
        measure.parse_attribution_stream(payload + b"x")


def test_actual_brotli_q11_and_zstd19_roundtrip() -> None:
    payload = (b"predictor-residual-" * 4096) + bytes(range(256))
    row = measure.compress_roundtrip(payload)
    assert row["raw_bytes"] == len(payload)
    assert row["brotli_q11"]["level"] == 11
    assert row["zstd_19"]["level"] == 19
    assert row["brotli_q11"]["bytes"] < len(payload)
    assert row["zstd_19"]["bytes"] < len(payload)
    assert row["decompression_verified"] is True


def test_measure_planes_exercises_all_production_modes_and_counted_attribution() -> None:
    y0, y1, labels, margins = _synthetic_planes()
    rows = measure.measure_planes(
        pair_ids=(7, 19),
        frame0_y_planes=y0,
        frame1_y_planes=y1,
        labels=labels,
        margins=margins,
    )
    assert [row["mode"] for row in rows] == list(measure.MODES)
    for row in rows:
        assert row["production_parseback_exact"] is True
        full = row["full_representation"]
        conditional = row["conditional_representation"]
        accounting = full["accounting"]
        assert full["production_counted_stream"]["bytes"] == (
            accounting["framing_bytes"]
            + accounting["bootstrap_brotli_q11_bytes"]
            + accounting["descriptor_bytes"]
            + accounting["residual_brotli_q11_bytes"]
        )
        assert conditional["production_brotli_q11_bytes"] == (
            accounting["descriptor_bytes"] + accounting["residual_brotli_q11_bytes"]
        )
        assert conditional["decoded_descriptor_plus_residual_bytes"] == (
            accounting["descriptor_bytes"] + accounting["decoded_residual_bytes"]
        )
        assert sum(bucket["selected_pixels"] for bucket in row["attribution"]["class"]) == 40
        assert sum(bucket["selected_pixels"] for bucket in row["attribution"]["margin"]) == 40
        for bucket in (*row["attribution"]["class"], *row["attribution"]["margin"]):
            assert bucket["counted_stream"]["decompression_verified"] is True


def test_measure_planes_refuses_uncovered_class_or_margin_values() -> None:
    y0, y1, labels, margins = _synthetic_planes()
    labels[0][0, 0] = 5
    with pytest.raises(measure.PredictorFloorError, match=r"classes 0\.\.4"):
        measure.measure_planes(
            pair_ids=(7, 19),
            frame0_y_planes=y0,
            frame1_y_planes=y1,
            labels=labels,
            margins=margins,
        )
    labels[0][0, 0] = 0
    margins[0][0, 0] = np.nan
    with pytest.raises(measure.PredictorFloorError, match="finite and nonnegative"):
        measure.measure_planes(
            pair_ids=(7, 19),
            frame0_y_planes=y0,
            frame1_y_planes=y1,
            labels=labels,
            margins=margins,
        )


def _counted_stream(raw: int, brotli_bytes: int, zstd_bytes: int) -> dict[str, Any]:
    return {
        "raw_bytes": raw,
        "brotli_q11": {"bytes": brotli_bytes},
        "zstd_19": {"bytes": zstd_bytes},
        "decompression_verified": True,
    }


def _chunk_doc(pair_ids: tuple[int, ...], mode_bias: int) -> dict[str, Any]:
    modes = []
    for mode_index, mode in enumerate(measure.MODES):
        attribution = {
            "class": [
                {"class_id": class_id, "selected_pixels": 10, "counted_stream": _counted_stream(20, 8, 7)}
                for class_id in range(5)
            ],
            "margin": [
                {
                    "margin_bin": name,
                    "selected_pixels": 9,
                    "counted_stream": _counted_stream(18, 7, 6),
                }
                for name in measure.MARGIN_NAMES
            ],
        }
        modes.append(
            {
                "mode": mode,
                "full_representation": {
                    "production_counted_stream": {
                        "bytes": mode_bias + mode_index,
                        "parseback_decompression_verified": True,
                    },
                    "secondary_double_compression_diagnostic": _counted_stream(100, 30 + mode_index, 20 + mode_index),
                },
                "conditional_representation": {
                    "production_brotli_q11_bytes": 12 + mode_index,
                    "decoded_descriptor_plus_residual_bytes": 80,
                    "direct_global_coder_ab": _counted_stream(80, 12 + mode_index, 10 + mode_index),
                },
                "prediction": {"residual_values": 1000, "residual_nonzero_values": 200, "residual_abs_sum": 300},
                "attribution": attribution,
            }
        )
    return {
        "schema": measure.SCHEMA_CHUNK,
        "pair_ids": list(pair_ids),
        "config": {"pair_ids": list(pair_ids), "cache_sha256": "a" * 64, "codec_sha256": "b" * 64},
        "modes": modes,
        "cache_custody": {"path": "/real/cache", "sha256": "a" * 64},
    }


def test_compose_requires_exact_four_chunks_and_sums_actual_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(measure, "_durable_path", lambda path, _field, **_kwargs: path.resolve())
    paths = []
    for index, chunk in enumerate(measure.CANONICAL_CHUNKS):
        path = tmp_path / f"chunk-{index}.json"
        path.write_text(json.dumps(_chunk_doc(chunk, 4)))
        paths.append(path)
    output = tmp_path / "n48.json"
    result = measure.compose_chunks(paths, output)
    assert result["pair_ids"] == list(range(48))
    assert result["best_predictor"]["mode"] == measure.MODES[0]
    assert result["modes"][0]["full_representation"]["production_brotli_q11_archive_section_bytes"] == 16
    assert result["modes"][0]["full_representation"]["production_stream_count"] == 4
    with pytest.raises(measure.PredictorFloorError, match="exactly four"):
        measure.compose_chunks(paths[:3], tmp_path / "bad.json")


def _small_fields(pair_count: int = 2) -> dict[str, np.ndarray]:
    frames = np.arange(pair_count * 4 * 4 * 3, dtype=np.uint8).reshape(pair_count, 4, 4, 3)
    return {
        "gt_f0": frames,
        "gt_f1": np.flip(frames, axis=2).copy(),
        "lstars": np.zeros((pair_count, 2, 2), dtype=np.uint8),
        "margins": np.ones((pair_count, 2, 2), dtype=np.float32),
        "gt_poses": np.zeros((pair_count, 6), dtype=np.float32),
    }


def test_measure_resume_rederives_and_matches_preserved_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fields = _small_fields()
    monkeypatch.setattr(measure, "CAMERA_HW", (4, 4))
    monkeypatch.setattr(measure, "SCORER_HW", (2, 2))
    monkeypatch.setattr(measure, "_durable_path", lambda path, _field, **_kwargs: path.resolve())
    monkeypatch.setattr(measure, "_load_cache", lambda *_args, **_kwargs: (fields, "c" * 64))
    monkeypatch.setattr(measure, "_tree_snapshot", lambda _path: {"exists": False})
    monkeypatch.setattr(measure, "_zstd_version", lambda _path: "zstd-test")
    monkeypatch.setattr(
        measure,
        "measure_planes",
        lambda **_kwargs: [{"mode": mode, "test_stub": True} for mode in measure.MODES],
    )
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"cache")
    base = {
        "output": tmp_path / "receipt.json",
        "state": tmp_path / "state.json",
        "stage_dir": tmp_path / "stages",
        "pairs": [0, 1],
        "chunk_index": None,
        "cache": cache,
        "sacred": tmp_path / "sacred",
        "zstd_binary": Path("zstd"),
        "allow_noncanonical_cache": True,
    }
    first = measure.measure_chunk(argparse.Namespace(**base, resume=False))
    stages_before = {path.name: path.read_bytes() for path in base["stage_dir"].iterdir()}
    base["output"].unlink()
    resumed = measure.measure_chunk(argparse.Namespace(**base, resume=True))
    stages_after = {path.name: path.read_bytes() for path in base["stage_dir"].iterdir()}
    assert first["pair_ids"] == resumed["pair_ids"] == [0, 1]
    assert stages_before == stages_after


def test_build_rung_e_inputs_uses_production_codec_parseback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fields = _small_fields()
    monkeypatch.setattr(measure, "CAMERA_HW", (4, 4))
    monkeypatch.setattr(measure, "SCORER_HW", (2, 2))
    monkeypatch.setattr(measure, "_load_cache", lambda *_args, **_kwargs: (fields, "d" * 64))
    result = measure.build_rung_e_inputs(
        tmp_path / "unused.npz",
        (0, 1),
        measure.AFFINE6_Q12_ID,
        require_canonical_hash=False,
    )
    assert result.frame0_y_planes.shape == (2, 2, 2, 3)
    assert result.frame1_y_planes.shape == (2, 2, 2, 3)
    assert all(len(descriptor) == 24 for descriptor in result.descriptors)
    assert len(result.predictor_payload_sha256) == 64


def test_score_inflated_raw_preserves_pair_order_and_aggregates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fields = _small_fields()
    monkeypatch.setattr(measure, "CAMERA_HW", (4, 4))
    monkeypatch.setattr(measure, "_load_cache", lambda *_args, **_kwargs: (fields, "e" * 64))
    observed: list[int] = []

    def fake_score(
        _segnet: object,
        _posenet: object,
        _torch: object,
        _frame0: np.ndarray,
        _frame1: np.ndarray,
        labels: np.ndarray,
        target_pose: np.ndarray,
    ) -> dict[str, Any]:
        pair_id = len(observed)
        observed.append(pair_id)
        assert labels.shape == (2, 2)
        assert target_pose.shape == (6,)
        return {"d_seg": pair_id / 10, "seg_mismatched_pixels": pair_id, "d_pose": pair_id / 20, "pose6": [0] * 6}

    monkeypatch.setattr(measure, "_score_one_pair", fake_score)
    raw = tmp_path / "inflated.raw"
    frames = fields["gt_f0"]
    raw.write_bytes(b"".join(frame.tobytes() for pair in zip(frames, frames, strict=True) for frame in pair))
    result = measure.score_inflated_raw(
        raw,
        pair_ids=(0, 1),
        cache_path=tmp_path / "unused.npz",
        require_canonical_hash=False,
        scorer_bundle=(object(), object(), object()),
    )
    assert observed == [0, 1]
    assert [row["pair_id"] for row in result["pairs"]] == [0, 1]
    assert result["mean_d_seg"] == pytest.approx(0.05)
    assert result["mean_d_pose"] == pytest.approx(0.025)


def test_ephemeral_policy_only_cleans_narrow_temp_tree(tmp_path: Path) -> None:
    target = tmp_path / "pact-rung-e-test"
    target.mkdir()
    (target / measure._EPHEMERAL_MARKER_NAME).write_bytes(measure._EPHEMERAL_MARKER_BYTES)
    (target / "archive.zip").write_bytes(b"ephemeral")
    measure._cleanup_ephemeral_output(target)
    assert not target.exists()

    unsafe = tmp_path / "unrelated-name"
    unsafe.mkdir()
    sentinel = unsafe / "preserve.bin"
    sentinel.write_bytes(b"keep")
    with pytest.raises(measure.PredictorFloorError, match="basename"):
        measure._cleanup_ephemeral_output(unsafe)
    assert sentinel.read_bytes() == b"keep"

    unowned = tmp_path / "pact-rung-e-unowned"
    unowned.mkdir()
    unowned_sentinel = unowned / "preserve.bin"
    unowned_sentinel.write_bytes(b"keep")
    with pytest.raises(measure.PredictorFloorError, match="ownership marker"):
        measure._cleanup_ephemeral_output(unowned)
    assert unowned_sentinel.read_bytes() == b"keep"

    with pytest.raises(measure.PredictorFloorError, match="system temporary root"):
        measure._ephemeral_output_root(Path("/Users/adpena/Projects/pact/pact-rung-e-unsafe"))


def test_rung_e_completion_seam_uses_production_completed_field(tmp_path: Path) -> None:
    raw_path = tmp_path / "inflated.raw"
    result = SimpleNamespace(completed=True, raw_path=raw_path)
    assert measure._completed_inflate_raw_path(result) == raw_path

    with pytest.raises(measure.PredictorFloorError, match="schema drift"):
        measure._completed_inflate_raw_path(SimpleNamespace(complete=True, raw_path=raw_path))
    with pytest.raises(measure.PredictorFloorError, match="did not complete"):
        measure._completed_inflate_raw_path(SimpleNamespace(completed=False, raw_path=raw_path))


def test_cli_has_measure_compose_and_rung_e_subcommands() -> None:
    parser = measure._parser()
    subcommands = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    assert set(subcommands.choices) == {"measure", "compose", "rung-e"}
