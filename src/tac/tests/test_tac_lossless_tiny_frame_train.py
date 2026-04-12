from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _frame(value: int) -> np.ndarray:
    return np.full((8, 16), value, dtype=np.int16)


def _write_record(tar: tarfile.TarFile, *, stem: str, frames: np.ndarray) -> None:
    token_buffer = io.BytesIO()
    np.save(token_buffer, frames, allow_pickle=False)
    token_payload = token_buffer.getvalue()
    json_payload = json.dumps({"file_name": f"{stem}.npy"}).encode("utf-8")

    token_info = tarfile.TarInfo(f"{stem}.token.npy")
    token_info.size = len(token_payload)
    tar.addfile(token_info, io.BytesIO(token_payload))

    json_info = tarfile.TarInfo(f"{stem}.json")
    json_info.size = len(json_payload)
    tar.addfile(json_info, io.BytesIO(json_payload))


class TacLosslessTinyFrameTrainTests(unittest.TestCase):
    def test_iter_local_commavq_tiny_frame_batches_pads_context_frames(self) -> None:
        from tac.lossless.tiny_frame_train import iter_local_commavq_tiny_frame_batches

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "data-0000.tar.gz"
            frames = np.stack([_frame(1), _frame(2), _frame(3)], axis=0)
            with tarfile.open(shard_path, "w:gz") as tar:
                _write_record(tar, stem="clip_a", frames=frames)

            batches = list(
                iter_local_commavq_tiny_frame_batches(
                    shard_paths=[shard_path],
                    batch_size=2,
                    context_frames=2,
                    max_records=1,
                    max_batches=1,
                )
            )

        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual(batch.context_frames.shape, (2, 2, 8, 16))
        self.assertEqual(batch.target_frames.shape, (2, 8, 16))
        self.assertTrue(np.array_equal(batch.context_frames[0, 0], np.zeros((8, 16), dtype=np.int16)))
        self.assertTrue(np.array_equal(batch.context_frames[0, 1], _frame(1)))
        self.assertTrue(np.array_equal(batch.target_frames[0], _frame(2)))
        self.assertTrue(np.array_equal(batch.context_frames[1, 0], _frame(1)))
        self.assertTrue(np.array_equal(batch.context_frames[1, 1], _frame(2)))
        self.assertTrue(np.array_equal(batch.target_frames[1], _frame(3)))
        self.assertEqual(batch.file_names, ("clip_a.npy", "clip_a.npy"))
        self.assertEqual(batch.target_frame_indices, (1, 2))

    def test_iter_local_commavq_tiny_frame_batches_resolves_cached_data_files(self) -> None:
        from tac.lossless.tiny_frame_train import iter_local_commavq_tiny_frame_batches

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_root = Path(tmpdir) / "hub" / "datasets--commaai--commavq" / "snapshots" / "snapshot-a"
            snapshot_root.mkdir(parents=True)
            shard_path = snapshot_root / "data-0000.tar.gz"
            with tarfile.open(shard_path, "w:gz") as tar:
                _write_record(tar, stem="clip_cached", frames=np.stack([_frame(4), _frame(5)], axis=0))

            with mock.patch.dict(os.environ, {"HF_HOME": tmpdir}, clear=False):
                batch = next(
                    iter_local_commavq_tiny_frame_batches(
                        data_files=["data-0000.tar.gz"],
                        batch_size=1,
                        context_frames=1,
                        max_records=1,
                        max_batches=1,
                    )
                )

        self.assertEqual(batch.file_names, ("clip_cached.npy",))
        self.assertEqual(batch.target_frame_indices, (1,))
        self.assertTrue(np.array_equal(batch.context_frames[0, 0], _frame(4)))
        self.assertTrue(np.array_equal(batch.target_frames[0], _frame(5)))

    def test_run_tiny_frame_supervised_step_reports_train_and_eval_summaries(self) -> None:
        import torch

        from tac.lossless.tiny_frame_predictor import TinyFramePredictorConfig
        from tac.lossless.tiny_frame_train import (
            TinyFrameSupervisedBatch,
            build_tiny_frame_training_model,
            run_tiny_frame_supervised_step,
        )

        batch = TinyFrameSupervisedBatch(
            context_frames=np.stack(
                [
                    np.stack([_frame(1), _frame(2)], axis=0),
                    np.stack([_frame(2), _frame(3)], axis=0),
                ],
                axis=0,
            ),
            target_frames=np.stack([_frame(3), _frame(4)], axis=0),
            file_names=("clip_a.npy", "clip_b.npy"),
            target_frame_indices=(2, 2),
        )
        torch.manual_seed(0)
        model = build_tiny_frame_training_model(
            TinyFramePredictorConfig(
                context_frames=2,
                positions=128,
                vocab_size=16,
                embed_dim=8,
                hidden_dim=16,
                mixer_layers=1,
            ),
            device="cpu",
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        train_before = model.output_projection.weight.detach().clone()

        train_summary = run_tiny_frame_supervised_step(model, batch, optimizer=optimizer, device="cpu")

        self.assertEqual(train_summary["mode"], "train")
        self.assertEqual(train_summary["batch_size"], 2)
        self.assertEqual(train_summary["token_count"], 256)
        self.assertGreater(train_summary["loss"], 0.0)
        self.assertGreaterEqual(train_summary["token_accuracy"], 0.0)
        self.assertLessEqual(train_summary["token_accuracy"], 1.0)
        self.assertGreaterEqual(train_summary["correct_tokens"], 0)
        self.assertLessEqual(train_summary["correct_tokens"], 256)
        self.assertFalse(torch.equal(train_before, model.output_projection.weight.detach()))

        eval_before = model.output_projection.weight.detach().clone()
        eval_summary = run_tiny_frame_supervised_step(model, batch, optimizer=None, device="cpu")

        self.assertEqual(eval_summary["mode"], "eval")
        self.assertEqual(eval_summary["batch_size"], 2)
        self.assertEqual(eval_summary["token_count"], 256)
        self.assertGreater(eval_summary["loss"], 0.0)
        self.assertTrue(torch.equal(eval_before, model.output_projection.weight.detach()))

    def test_probe_tiny_frame_training_writes_json_artifact_with_model_size_metadata(self) -> None:
        import torch

        from tac.lossless.tiny_frame_train import (
            build_tiny_frame_training_model,
            probe_tiny_frame_training,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_path = root / "data-0000.tar.gz"
            output_path = root / "probe.json"
            frames = np.stack([_frame(1), _frame(2), _frame(3)], axis=0)
            with tarfile.open(shard_path, "w:gz") as tar:
                _write_record(tar, stem="clip_probe", frames=frames)

            torch.manual_seed(0)
            expected_model = build_tiny_frame_training_model(
                "tiny_frame_predictor_small",
                context_frames=2,
                device="cpu",
            )
            expected_parameter_count = sum(int(param.numel()) for param in expected_model.parameters())

            torch.manual_seed(0)
            result = probe_tiny_frame_training(
                profile="tiny_frame_predictor_small",
                output_path=output_path,
                shard_paths=[shard_path],
                batch_size=2,
                context_frames=2,
                max_records=1,
                sample_offset=0,
                max_batches=1,
                learning_rate=0.05,
                device="cpu",
            )

            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(result["command"], "lossless_tiny_frame_train_probe")
        self.assertEqual(result["profile"], "tiny_frame_predictor_small")
        self.assertEqual(result["output_path"], str(output_path))
        self.assertEqual(result["parameter_count"], expected_parameter_count)
        self.assertIsInstance(result["state_dict_bytes"], int)
        self.assertGreater(result["state_dict_bytes"], 0)
        self.assertEqual(result["observed_batch_count"], 1)
        self.assertEqual(result["file_names"], ["clip_probe.npy", "clip_probe.npy"])
        self.assertEqual(result["target_frame_indices"], [1, 2])
        self.assertEqual(result["train"]["mode"], "train")
        self.assertEqual(result["eval"]["mode"], "eval")
        self.assertEqual(written, result)


if __name__ == "__main__":
    unittest.main()
