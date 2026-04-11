from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessTokenRgbBridgeTests(unittest.TestCase):
    def test_resolve_onnx_execution_providers_prefers_coreml_then_cpu(self) -> None:
        from tac.lossless.token_rgb_bridge import resolve_onnx_execution_providers

        providers = resolve_onnx_execution_providers(
            ["AzureExecutionProvider", "CPUExecutionProvider", "CoreMLExecutionProvider"]
        )

        self.assertEqual(providers, ["CoreMLExecutionProvider", "CPUExecutionProvider"])

    def test_load_official_commavq_onnx_bridge_prefers_coreml_then_cpu(self) -> None:
        from tac.lossless import token_rgb_bridge as bridge

        fake_session = mock.Mock()
        fake_session.get_inputs.return_value = [SimpleNamespace(name="tokens")]
        fake_session.run.return_value = [np.ones((2, 3, 2, 2), dtype=np.float32)]
        fake_ort = SimpleNamespace(
            get_available_providers=lambda: [
                "AzureExecutionProvider",
                "CPUExecutionProvider",
                "CoreMLExecutionProvider",
            ],
            InferenceSession=mock.Mock(return_value=fake_session),
        )

        with mock.patch.object(bridge, "_require_onnxruntime", return_value=fake_ort):
            with mock.patch.object(
                bridge,
                "ensure_official_decoder_onnx_path",
                return_value=Path("/tmp/decoder.onnx"),
            ):
                decoder, transpose_and_clip_fn, metadata = bridge.load_official_commavq_onnx_bridge()

        decoded = decoder(np.arange(2 * 128, dtype=np.int64).reshape(2, 128))

        fake_ort.InferenceSession.assert_called_once_with(
            "/tmp/decoder.onnx",
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        fake_session.run.assert_called_once()
        self.assertEqual(
            fake_session.run.call_args.args[1]["tokens"].tolist(),
            np.arange(2 * 128, dtype=np.int64).reshape(2, 128).tolist(),
        )
        self.assertIs(transpose_and_clip_fn, bridge.canonical_transpose_and_clip)
        self.assertEqual(metadata["bridge_backend"], "onnx")
        self.assertEqual(metadata["execution_provider"], "CoreMLExecutionProvider")
        self.assertEqual(metadata["execution_providers"], ["CoreMLExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(decoded.shape, (2, 3, 2, 2))

    def test_load_official_commavq_bridge_falls_back_to_torch_when_onnx_unavailable(self) -> None:
        from tac.lossless import token_rgb_bridge as bridge

        expected_decoder = object()
        expected_metadata = {
            "bridge_backend": "torch",
            "decoder_artifact_url": bridge.OFFICIAL_TORCH_DECODER_URL,
        }

        with mock.patch.object(
            bridge,
            "load_official_commavq_onnx_bridge",
            side_effect=ImportError("onnxruntime missing"),
        ):
            with mock.patch.object(
                bridge,
                "load_official_commavq_torch_bridge",
                return_value=(expected_decoder, bridge.canonical_transpose_and_clip, expected_metadata),
            ) as mocked_torch:
                decoder, transpose_and_clip_fn, metadata = bridge.load_official_commavq_bridge(device="cpu")

        mocked_torch.assert_called_once_with(
            device="cpu",
            dtype="auto",
            commavq_root=None,
            decoder_url=bridge.OFFICIAL_TORCH_DECODER_URL,
        )
        self.assertIs(decoder, expected_decoder)
        self.assertIs(transpose_and_clip_fn, bridge.canonical_transpose_and_clip)
        self.assertEqual(metadata["bridge_backend"], "torch")
        self.assertIn("onnxruntime missing", metadata["bridge_fallback_reason"])

    def test_resolve_bridge_dtype_name_prefers_float32_for_official_decoder(self) -> None:
        from tac.lossless.token_rgb_bridge import resolve_bridge_dtype_name

        self.assertEqual(resolve_bridge_dtype_name(device="mps", dtype="auto"), "float32")
        self.assertEqual(resolve_bridge_dtype_name(device="mps", dtype="float16"), "float32")
        self.assertEqual(resolve_bridge_dtype_name(device="cpu", dtype="float32"), "float32")

    def test_canonical_transpose_and_clip_matches_official_layout(self) -> None:
        from tac.lossless.token_rgb_bridge import canonical_transpose_and_clip

        tensor = np.array(
            [
                [
                    [[-1.0, 1.0], [2.0, 260.0]],
                    [[3.0, 4.0], [5.0, 6.0]],
                    [[7.0, 8.0], [9.0, 10.0]],
                ]
            ],
            dtype=np.float32,
        )

        frames = canonical_transpose_and_clip(tensor)

        self.assertEqual(frames.shape, (1, 2, 2, 3))
        self.assertEqual(frames.dtype, np.uint8)
        self.assertEqual(frames[0, 0, 0].tolist(), [0, 3, 7])

    def test_resolve_commavq_official_root_accepts_explicit_canonical_checkout(self) -> None:
        from tac.lossless.token_rgb_bridge import resolve_commavq_official_root

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "utils").mkdir()
            (root / "notebooks").mkdir()
            (root / "utils" / "vqvae.py").write_text("class Decoder: pass\nclass CompressorConfig: pass\n")
            (root / "utils" / "video.py").write_text("def transpose_and_clip(x): return x\n")
            (root / "notebooks" / "decode.ipynb").write_text("{}\n")

            resolved = resolve_commavq_official_root(root)

        self.assertEqual(resolved, root)

    def test_decode_commavq_tokens_to_rgb_frames_batches_through_decoder(self) -> None:
        from tac.lossless.token_rgb_bridge import decode_commavq_tokens_to_rgb_frames

        seen_batches: list[list[list[int]]] = []

        class FakeDecoder:
            def __call__(self, batch):
                arr = np.asarray(batch)
                seen_batches.append([[int(item) for item in row.tolist()] for row in arr])
                bsz = arr.shape[0]
                out = np.zeros((bsz, 3, 2, 2), dtype=np.float32)
                for index in range(bsz):
                    out[index] += float(index + 1)
                return out

        def fake_transpose_and_clip(arr):
            arr = np.asarray(arr)
            return np.transpose(arr, (0, 2, 3, 1)).astype(np.uint8)

        tokens = np.arange(3 * 8 * 16, dtype=np.int16).reshape(3, 8, 16)

        frames = decode_commavq_tokens_to_rgb_frames(
            tokens,
            decoder=FakeDecoder(),
            transpose_and_clip_fn=fake_transpose_and_clip,
            batch_size=2,
            device="cpu",
        )

        self.assertEqual(seen_batches, [
            np.arange(2 * 8 * 16, dtype=np.int16).reshape(2, 8 * 16).tolist(),
            np.arange(2 * 8 * 16, 3 * 8 * 16, dtype=np.int16).reshape(1, 8 * 16).tolist(),
        ])
        self.assertEqual(frames.shape, (3, 2, 2, 3))
        self.assertEqual(frames.dtype, np.uint8)

    def test_decode_commavq_tokens_to_rgb_frames_preserves_cube_shape_for_numpy_bridge(self) -> None:
        from tac.lossless.token_rgb_bridge import decode_commavq_tokens_to_rgb_frames

        seen_shapes: list[tuple[int, ...]] = []

        class FakeOnnxDecoder:
            _tac_input_kind = "numpy"

            def __call__(self, batch):
                arr = np.asarray(batch)
                seen_shapes.append(arr.shape)
                return np.zeros((arr.shape[0], 3, 2, 2), dtype=np.float32)

        frames = decode_commavq_tokens_to_rgb_frames(
            np.arange(3 * 8 * 16, dtype=np.int16).reshape(3, 8, 16),
            decoder=FakeOnnxDecoder(),
            transpose_and_clip_fn=lambda arr: np.transpose(np.asarray(arr), (0, 2, 3, 1)).astype(np.uint8),
            batch_size=2,
            device="cpu",
        )

        self.assertEqual(seen_shapes, [(2, 8, 16), (1, 8, 16)])
        self.assertEqual(frames.shape, (3, 2, 2, 3))

    def test_decode_commavq_tokens_to_rgb_frames_keeps_numpy_batches_for_onnx_bridge(self) -> None:
        from tac.lossless.token_rgb_bridge import decode_commavq_tokens_to_rgb_frames

        seen_batch_types: list[type] = []

        class FakeOnnxDecoder:
            _tac_input_kind = "numpy"

            def __call__(self, batch):
                seen_batch_types.append(type(batch))
                arr = np.asarray(batch)
                return np.ones((arr.shape[0], 3, 2, 2), dtype=np.float32) * 11.0

        frames = decode_commavq_tokens_to_rgb_frames(
            np.arange(3 * 8 * 16, dtype=np.int16).reshape(3, 8, 16),
            decoder=FakeOnnxDecoder(),
            transpose_and_clip_fn=lambda arr: np.transpose(np.asarray(arr), (0, 2, 3, 1)).astype(np.uint8),
            batch_size=2,
            device="mps",
        )

        self.assertEqual(seen_batch_types, [np.ndarray, np.ndarray])
        self.assertEqual(frames.shape, (3, 2, 2, 3))

    def test_decode_commavq_token_file_to_rgb_uses_injected_bridge_loader(self) -> None:
        from tac.lossless.token_rgb_bridge import decode_commavq_token_file_to_rgb

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.npy"
            output_path = root / "frames.npy"
            np.save(token_path, np.arange(4 * 8 * 16, dtype=np.int16).reshape(4, 8, 16))

            class FakeDecoder:
                def __call__(self, batch):
                    arr = np.asarray(batch)
                    bsz = arr.shape[0]
                    out = np.ones((bsz, 3, 2, 2), dtype=np.float32) * 7.0
                    return out

            def fake_loader(**_kwargs):
                return (
                    FakeDecoder(),
                    lambda arr: np.transpose(np.asarray(arr), (0, 2, 3, 1)).astype(np.uint8),
                    {
                        "bridge_backend": "onnx",
                        "execution_provider": "CPUExecutionProvider",
                    },
                )

            result = decode_commavq_token_file_to_rgb(
                token_path=token_path,
                output_path=output_path,
                max_frames=3,
                batch_size=2,
                device="cpu",
                bridge_loader=fake_loader,
            )

            frames = np.load(output_path, mmap_mode="r")

        self.assertEqual(result["command"], "lossless_token_rgb_sample")
        self.assertEqual(result["frame_count"], 3)
        self.assertEqual(result["output_path"], str(output_path))
        self.assertEqual(result["bridge_backend"], "onnx")
        self.assertEqual(result["execution_provider"], "CPUExecutionProvider")
        self.assertEqual(tuple(frames.shape), (3, 2, 2, 3))
        self.assertEqual(frames.dtype, np.uint8)

    def test_decode_commavq_token_file_to_rgb_rejects_non_token_cube(self) -> None:
        from tac.lossless.token_rgb_bridge import decode_commavq_token_file_to_rgb

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.npy"
            output_path = root / "frames.npy"
            np.save(token_path, np.arange(32, dtype=np.int16))

            with self.assertRaisesRegex(ValueError, "shape"):
                decode_commavq_token_file_to_rgb(
                    token_path=token_path,
                    output_path=output_path,
                    bridge_loader=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not load bridge")),
                )

    def test_decode_commavq_token_file_to_rgb_accepts_flat_128_token_frames(self) -> None:
        from tac.lossless.token_rgb_bridge import decode_commavq_token_file_to_rgb

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.npy"
            output_path = root / "frames.npy"
            np.save(token_path, np.arange(4 * 128, dtype=np.int16).reshape(4, 128))

            class FakeDecoder:
                def __call__(self, batch):
                    arr = np.asarray(batch)
                    return np.ones((arr.shape[0], 3, 2, 2), dtype=np.float32) * 5.0

            result = decode_commavq_token_file_to_rgb(
                token_path=token_path,
                output_path=output_path,
                batch_size=2,
                device="cpu",
                bridge_loader=lambda **_kwargs: (
                    FakeDecoder(),
                    lambda arr: np.transpose(np.asarray(arr), (0, 2, 3, 1)).astype(np.uint8),
                ),
            )

            frames = np.load(output_path, mmap_mode="r")

        self.assertEqual(result["frame_count"], 4)
        self.assertEqual(tuple(frames.shape), (4, 2, 2, 3))


if __name__ == "__main__":
    unittest.main()
