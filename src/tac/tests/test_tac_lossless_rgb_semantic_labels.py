from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _outdoor_frames(*, frame_count: int) -> np.ndarray:
    frames = np.zeros((frame_count, 6, 6, 3), dtype=np.uint8)
    frames[:, :3, :, :] = np.array([80, 160, 235], dtype=np.uint8)
    frames[:, 3:, :, :] = np.array([72, 72, 72], dtype=np.uint8)
    frames[:, 4:, 2:4, :] = np.array([220, 220, 220], dtype=np.uint8)
    return frames


def _warm_indoor_frames(*, frame_count: int) -> np.ndarray:
    frames = np.zeros((frame_count, 6, 6, 3), dtype=np.uint8)
    frames[:, :, :, :] = np.array([210, 150, 90], dtype=np.uint8)
    frames[:, :, 3:, :] = np.array([190, 120, 70], dtype=np.uint8)
    return frames


class TacLosslessRgbSemanticLabelsTests(unittest.TestCase):
    def test_rgb_semantic_label_tuple_separates_same_regime_but_different_brightness(self) -> None:
        from tac.lossless.rgb_semantic_labels import rgb_semantic_label_tuple

        dim_scene = np.full((4, 6, 6, 3), 90, dtype=np.uint8)
        bright_scene = np.full((4, 6, 6, 3), 105, dtype=np.uint8)

        dim_label = rgb_semantic_label_tuple(dim_scene)
        bright_label = rgb_semantic_label_tuple(bright_scene)

        self.assertNotEqual(dim_label, bright_label)
        self.assertNotEqual(dim_label[0], bright_label[0])

    def test_rgb_semantic_label_tuple_detects_sky_and_road_scene_proxies(self) -> None:
        from tac.lossless.rgb_semantic_labels import rgb_semantic_label_tuple

        label = rgb_semantic_label_tuple(_outdoor_frames(frame_count=4))

        self.assertEqual(len(label), 8)
        self.assertEqual(label[3], 0)  # cool scene
        self.assertEqual(label[4], 3)  # strong sky proxy
        self.assertEqual(label[5], 3)  # strong road proxy
        self.assertEqual(label[7], 0)  # temporally stable

    def test_rgb_semantic_label_tuple_is_deterministic_across_numeric_dtypes(self) -> None:
        from tac.lossless.rgb_semantic_labels import rgb_semantic_label_tuple

        frames = _warm_indoor_frames(frame_count=9)
        label_uint8 = rgb_semantic_label_tuple(frames, max_keyframes=3)
        label_float32 = rgb_semantic_label_tuple(frames.astype(np.float32), max_keyframes=3)

        self.assertEqual(label_uint8, label_float32)
        self.assertEqual(label_uint8[3], 4)  # warm scene
        self.assertEqual(label_uint8[4], 0)  # no sky proxy
        self.assertEqual(label_uint8[5], 0)  # no road proxy

    def test_extract_rgb_semantic_label_from_tokens_matches_direct_frame_labels(self) -> None:
        from tac.lossless.rgb_semantic_labels import (
            extract_rgb_semantic_label_from_tokens,
            rgb_semantic_label_tuple,
        )

        frames = _outdoor_frames(frame_count=5)
        expected = rgb_semantic_label_tuple(frames, max_keyframes=3)
        bridge_calls: list[dict[str, object]] = []
        cursor = {"start": 0}

        class FakeDecoder:
            def __call__(self, batch):
                arr = np.asarray(batch)
                start = cursor["start"]
                stop = start + arr.shape[0]
                cursor["start"] = stop
                return np.transpose(frames[start:stop].astype(np.float32), (0, 3, 1, 2))

        def fake_bridge_loader(**kwargs):
            bridge_calls.append(dict(kwargs))
            cursor["start"] = 0
            return (
                FakeDecoder(),
                lambda arr: np.transpose(np.asarray(arr), (0, 2, 3, 1)).astype(np.uint8),
                {"bridge_backend": "fake"},
            )

        label = extract_rgb_semantic_label_from_tokens(
            np.arange(5 * 8 * 16, dtype=np.int16).reshape(5, 8, 16),
            bridge_loader=fake_bridge_loader,
            batch_size=2,
            max_keyframes=3,
            device="cpu",
        )

        self.assertEqual(label, expected)
        self.assertEqual(len(bridge_calls), 1)
        self.assertEqual(bridge_calls[0]["device"], "cpu")
        self.assertEqual(bridge_calls[0]["dtype"], "auto")

    def test_extract_rgb_semantic_label_from_tokens_decodes_only_sampled_keyframes(self) -> None:
        from tac.lossless.rgb_semantic_labels import (
            extract_rgb_semantic_label_from_tokens,
            rgb_semantic_label_tuple,
        )

        frames = _outdoor_frames(frame_count=9)
        expected = rgb_semantic_label_tuple(frames, max_keyframes=3)
        seen_batches: list[tuple[int, ...]] = []
        seen_token_rows: list[list[int]] = []

        class FakeDecoder:
            _tac_input_kind = "numpy"

            def __call__(self, batch):
                arr = np.asarray(batch)
                seen_batches.append(arr.shape)
                seen_token_rows.extend(arr[:, 0, :].tolist())
                selected_frames = frames[arr[:, 0, 0].astype(np.int64)]
                return np.transpose(selected_frames.astype(np.float32), (0, 3, 1, 2))

        def fake_bridge_loader(**_kwargs):
            return (
                FakeDecoder(),
                lambda arr: np.transpose(np.asarray(arr), (0, 2, 3, 1)).astype(np.uint8),
                {"bridge_backend": "fake"},
            )

        tokens = np.repeat(np.arange(9, dtype=np.int16)[:, np.newaxis, np.newaxis], 8 * 16, axis=1).reshape(9, 8, 16)
        label = extract_rgb_semantic_label_from_tokens(
            tokens,
            bridge_loader=fake_bridge_loader,
            batch_size=8,
            max_keyframes=3,
            device="cpu",
        )

        self.assertEqual(label, expected)
        self.assertEqual(seen_batches, [(3, 8, 16)])
        self.assertEqual([row[0] for row in seen_token_rows], [0, 4, 8])

    def test_build_rgb_label_map_sample_reuses_bridge_across_records(self) -> None:
        from tac.lossless.rgb_semantic_labels import build_rgb_label_map_sample

        examples = [
            {
                "json": {"file_name": "clip_b.npy"},
                "token.npy": np.ones((1, 8, 16), dtype=np.int16),
            },
            {
                "json": {"file_name": "clip_a.npy"},
                "token.npy": np.zeros((1, 8, 16), dtype=np.int16),
            },
        ]
        bridge_calls: list[dict[str, object]] = []
        decoder_batches: list[tuple[int, ...]] = []
        def fake_dataset_loader(_dataset_name: str, *, num_proc=None, data_files=None, streaming=None):
            return {"train": examples}

        class FakeDecoder:
            _tac_input_kind = "numpy"

            def __call__(self, batch):
                arr = np.asarray(batch)
                decoder_batches.append(arr.shape)
                out = np.zeros((arr.shape[0], 3, 6, 6), dtype=np.float32)
                for index in range(arr.shape[0]):
                    if int(arr[index].sum()) == 0:
                        frame = _outdoor_frames(frame_count=1)[0]
                    else:
                        frame = _warm_indoor_frames(frame_count=1)[0]
                    out[index] = np.transpose(frame.astype(np.float32), (2, 0, 1))
                return out

        def fake_bridge_loader(**kwargs):
            bridge_calls.append(dict(kwargs))
            return (
                FakeDecoder(),
                lambda arr: (_ for _ in ()).throw(AssertionError("build_rgb_label_map_sample should not transpose RGB frames")),
                {"bridge_backend": "fake"},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rgb_labels.json"
            with mock.patch("tac.lossless.rgb_semantic_labels.resolve_local_commavq_cached_data_files", return_value=None):
                result = build_rgb_label_map_sample(
                    output_path=output_path,
                    split=[0, 1],
                    max_records=2,
                    dataset_loader=fake_dataset_loader,
                    bridge_loader=fake_bridge_loader,
                    batch_size=8,
                    device="cpu",
                )

            payload = json.loads(output_path.read_text())

        self.assertEqual(len(bridge_calls), 1)
        self.assertEqual(decoder_batches, [(2, 8, 16)])
        self.assertEqual(result["command"], "lossless_rgb_labels_sample")
        self.assertEqual(result["record_count"], 2)
        self.assertTrue(result["local_only"])
        self.assertEqual(sorted(payload), ["clip_a.npy", "clip_b.npy"])
        self.assertEqual(payload["clip_a.npy"][4], 3)
        self.assertEqual(payload["clip_b.npy"][3], 4)

    def test_build_rgb_label_map_sample_prefers_single_pass_iteration_over_indexing(self) -> None:
        from tac.lossless.rgb_semantic_labels import build_rgb_label_map_sample

        examples = [
            {
                "json": {"file_name": "clip_a.npy"},
                "token.npy": np.zeros((1, 8, 16), dtype=np.int16),
            },
            {
                "json": {"file_name": "clip_b.npy"},
                "token.npy": np.ones((1, 8, 16), dtype=np.int16),
            },
        ]

        class FakeTrain:
            def __len__(self):
                return len(examples)

            def __iter__(self):
                return iter(examples)

            def __getitem__(self, index):
                raise AssertionError("build_rgb_label_map_sample should not use indexed dataset access")

        def fake_dataset_loader(_dataset_name: str, *, num_proc=None, data_files=None, streaming=None):
            return {"train": FakeTrain()}

        class FakeDecoder:
            _tac_input_kind = "numpy"

            def __call__(self, batch):
                arr = np.asarray(batch)
                out = np.zeros((arr.shape[0], 3, 6, 6), dtype=np.float32)
                for index in range(arr.shape[0]):
                    if int(arr[index].sum()) == 0:
                        frame = _outdoor_frames(frame_count=1)[0]
                    else:
                        frame = _warm_indoor_frames(frame_count=1)[0]
                    out[index] = np.transpose(frame.astype(np.float32), (2, 0, 1))
                return out

        def fake_bridge_loader(**_kwargs):
            return (
                FakeDecoder(),
                lambda arr: np.transpose(np.asarray(arr), (0, 2, 3, 1)).astype(np.uint8),
                {"bridge_backend": "fake"},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rgb_labels.json"
            with mock.patch("tac.lossless.rgb_semantic_labels.resolve_local_commavq_cached_data_files", return_value=None):
                result = build_rgb_label_map_sample(
                    output_path=output_path,
                    split=[0, 1],
                    max_records=2,
                    dataset_loader=fake_dataset_loader,
                    bridge_loader=fake_bridge_loader,
                    batch_size=8,
                    device="cpu",
                )

            payload = json.loads(output_path.read_text())

        self.assertEqual(result["record_count"], 2)
        self.assertEqual(sorted(payload), ["clip_a.npy", "clip_b.npy"])

    def test_build_rgb_label_map_sample_requests_streaming_dataset_load(self) -> None:
        from tac.lossless import rgb_semantic_labels as module

        calls: list[dict[str, object]] = []
        examples = [
            {
                "json": {"file_name": "clip_a.npy"},
                "token.npy": np.zeros((1, 8, 16), dtype=np.int16),
            }
        ]

        class FakeDecoder:
            _tac_input_kind = "numpy"

            def __call__(self, batch):
                arr = np.asarray(batch)
                out = np.zeros((arr.shape[0], 3, 6, 6), dtype=np.float32)
                out[0] = np.transpose(_outdoor_frames(frame_count=1)[0].astype(np.float32), (2, 0, 1))
                return out

        def fake_load_commavq_dataset(*, split, dataset_loader=None, dataset_name=None, num_proc=None, streaming=None):
            calls.append(
                {
                    "split": split,
                    "dataset_loader": dataset_loader,
                    "dataset_name": dataset_name,
                    "num_proc": num_proc,
                    "streaming": streaming,
                }
            )
            return {"train": examples}

        def fake_bridge_loader(**_kwargs):
            return (
                FakeDecoder(),
                lambda arr: np.transpose(np.asarray(arr), (0, 2, 3, 1)).astype(np.uint8),
                {"bridge_backend": "fake"},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rgb_labels.json"
            with mock.patch.object(module, "load_commavq_dataset", side_effect=fake_load_commavq_dataset):
                with mock.patch.object(module, "resolve_local_commavq_cached_data_files", return_value=None):
                    module.build_rgb_label_map_sample(
                        output_path=output_path,
                        split=[0, 1],
                        max_records=1,
                        bridge_loader=fake_bridge_loader,
                        batch_size=8,
                        device="cpu",
                    )

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["streaming"])

    def test_build_rgb_label_map_sample_reuses_existing_output_entries(self) -> None:
        from tac.lossless.rgb_semantic_labels import build_rgb_label_map_sample

        examples = [
            {
                "json": {"file_name": "clip_a.npy"},
                "token.npy": np.zeros((1, 8, 16), dtype=np.int16),
            },
            {
                "json": {"file_name": "clip_b.npy"},
                "token.npy": np.ones((1, 8, 16), dtype=np.int16),
            },
        ]
        decoder_batches: list[tuple[int, ...]] = []

        def fake_dataset_loader(_dataset_name: str, *, num_proc=None, data_files=None, streaming=None):
            return {"train": examples}

        class FakeDecoder:
            _tac_input_kind = "numpy"

            def __call__(self, batch):
                arr = np.asarray(batch)
                decoder_batches.append(arr.shape)
                out = np.zeros((arr.shape[0], 3, 6, 6), dtype=np.float32)
                out[0] = np.transpose(_warm_indoor_frames(frame_count=1)[0].astype(np.float32), (2, 0, 1))
                return out

        def fake_bridge_loader(**_kwargs):
            return (
                FakeDecoder(),
                lambda arr: np.transpose(np.asarray(arr), (0, 2, 3, 1)).astype(np.uint8),
                {"bridge_backend": "fake"},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rgb_labels.json"
            output_path.write_text(json.dumps({"clip_a.npy": [10, 0, 0, 2, 0, 3, 0, 0]}), encoding="utf-8")

            with mock.patch("tac.lossless.rgb_semantic_labels.resolve_local_commavq_cached_data_files", return_value=None):
                result = build_rgb_label_map_sample(
                    output_path=output_path,
                    split=[0, 1],
                    max_records=2,
                    dataset_loader=fake_dataset_loader,
                    bridge_loader=fake_bridge_loader,
                    batch_size=8,
                    device="cpu",
                )

            payload = json.loads(output_path.read_text())

        self.assertEqual(result["record_count"], 2)
        self.assertEqual(decoder_batches, [(1, 8, 16)])
        self.assertIn("clip_a.npy", payload)
        self.assertIn("clip_b.npy", payload)

    def test_build_rgb_label_map_sample_prefers_local_cached_shards_over_dataset_loader(self) -> None:
        from tac.lossless import rgb_semantic_labels as module

        examples = [
            {
                "json": {"file_name": "clip_a.npy"},
                "token.npy": np.zeros((1, 8, 16), dtype=np.int16),
            }
        ]

        class FakeDecoder:
            _tac_input_kind = "numpy"

            def __call__(self, batch):
                arr = np.asarray(batch)
                out = np.zeros((arr.shape[0], 3, 6, 6), dtype=np.float32)
                out[0] = np.transpose(_outdoor_frames(frame_count=1)[0].astype(np.float32), (2, 0, 1))
                return out

        def fake_bridge_loader(**_kwargs):
            return (
                FakeDecoder(),
                lambda arr: np.transpose(np.asarray(arr), (0, 2, 3, 1)).astype(np.uint8),
                {"bridge_backend": "fake"},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rgb_labels.json"
            with mock.patch.object(module, "resolve_local_commavq_cached_data_files", return_value=["/tmp/data-0000.tar.gz"]):
                with mock.patch.object(module, "load_local_commavq_record_sample", return_value=examples) as mocked_local:
                    with mock.patch.object(
                        module,
                        "load_commavq_dataset",
                        side_effect=AssertionError("should not call load_commavq_dataset when local cached shards exist"),
                    ):
                        result = module.build_rgb_label_map_sample(
                            output_path=output_path,
                            split=[0, 1],
                            max_records=1,
                            bridge_loader=fake_bridge_loader,
                            batch_size=8,
                            device="cpu",
                        )

            payload = json.loads(output_path.read_text())

        mocked_local.assert_called_once_with(["/tmp/data-0000.tar.gz"], max_records=1)
        self.assertEqual(result["record_count"], 1)
        self.assertEqual(sorted(payload), ["clip_a.npy"])


if __name__ == "__main__":
    unittest.main()
