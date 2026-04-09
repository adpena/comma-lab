from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from contextlib import nullcontext

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "train_postfilter_qat_ema.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_postfilter_qat_ema", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TrainPostfilterQatEmaTests(unittest.TestCase):
    def test_fake_quant_backward_handles_zero_scale_tensor(self) -> None:
        mod = load_module()
        weight = torch.zeros(4, requires_grad=True)
        quantized = mod.fake_quant(weight)
        loss = quantized.sum()
        loss.backward()
        self.assertTrue(torch.equal(weight.grad, torch.ones_like(weight)))

    def test_maybe_transfer_pairs_to_device_keeps_cpu_pairs_by_default(self) -> None:
        mod = load_module()
        pairs = [torch.zeros(1, 2, 4, 4, 3), torch.ones(1, 2, 4, 4, 3)]

        kept = mod.maybe_transfer_pairs_to_device(pairs, torch.device("cpu"), eager=False)

        self.assertIs(kept, pairs)
        self.assertEqual(str(kept[0].device), "cpu")

    def test_maybe_transfer_pairs_to_device_moves_when_eager(self) -> None:
        mod = load_module()
        pairs = [torch.zeros(1, 2, 4, 4, 3)]

        moved = mod.maybe_transfer_pairs_to_device(pairs, torch.device("cpu"), eager=True)

        self.assertIsNot(moved, pairs)
        self.assertEqual(str(moved[0].device), "cpu")

    def test_build_pair_start_indices_matches_seq_len_stride(self) -> None:
        mod = load_module()
        self.assertEqual(mod.build_pair_start_indices(5, 2), [0, 2])
        self.assertEqual(mod.build_pair_start_indices(4, 2), [0, 2])
        self.assertEqual(mod.build_pair_start_indices(1, 2), [])

    def test_pair_from_frames_matches_old_pair_shape(self) -> None:
        mod = load_module()
        frames = [torch.full((4, 4, 3), fill_value=i, dtype=torch.uint8) for i in range(4)]
        pair = mod.pair_from_frames(frames, 2)
        self.assertEqual(tuple(pair.shape), (1, 2, 4, 4, 3))
        self.assertEqual(int(pair[0, 0, 0, 0, 0]), 2)
        self.assertEqual(int(pair[0, 1, 0, 0, 0]), 3)

    def test_saliency_pair_at_repeats_last_frame_when_needed(self) -> None:
        mod = load_module()
        base = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
            ],
            dtype=torch.float32,
        )
        pair = mod.saliency_pair_at(base, start_idx=4, alpha=2.0, device=torch.device("cpu"))
        self.assertEqual(tuple(pair.shape), (2, 1, 2, 2))
        self.assertTrue(torch.equal(pair[0, 0], torch.tensor([[3.0, 5.0], [7.0, 9.0]])))
        self.assertTrue(torch.equal(pair[1, 0], torch.tensor([[3.0, 5.0], [7.0, 9.0]])))

    def test_save_best_checkpoint_writes_fp32_and_int8_artifacts(self) -> None:
        mod = load_module()
        model = mod.QATPostFilter(hidden=4, kernel=3)
        ema = mod.EMA(model, decay=0.9)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            result = mod.save_best_checkpoint(
                model=model,
                ema=ema,
                output_dir=out_dir,
                tag="unit_qat",
                meta={"variant": "residual", "hidden": 4, "kernel": 3},
                epoch=12,
                scorer=1.23,
            )
            fp32_path = out_dir / "postfilter_unit_qat_best_fp32.pt"
            int8_path = out_dir / "postfilter_unit_qat_best_int8.pt"
            meta_path = out_dir / "postfilter_unit_qat_best_meta.json"

            self.assertTrue(fp32_path.exists())
            self.assertTrue(int8_path.exists())
            self.assertTrue(meta_path.exists())
            self.assertEqual(result["epoch"], 12)
            self.assertEqual(result["scorer"], 1.23)

    def test_save_best_checkpoint_supports_per_channel_int8(self) -> None:
        mod = load_module()
        model = mod.QATPostFilter(hidden=4, kernel=3)
        ema = mod.EMA(model, decay=0.9)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            mod.save_best_checkpoint(
                model=model,
                ema=ema,
                output_dir=out_dir,
                tag="unit_qat_pc",
                meta={"variant": "residual", "hidden": 4, "kernel": 3},
                epoch=3,
                scorer=1.11,
                per_channel_int8=True,
            )

            saved = torch.load(out_dir / "postfilter_unit_qat_pc_best_int8.pt", weights_only=True)
            self.assertEqual(tuple(saved["conv1.weight.s"].shape), (4,))
            self.assertEqual(saved["conv1.weight.q"].dtype, torch.int8)
            self.assertEqual(saved["conv1.bias"].dtype, torch.float32)

    def test_quantize_state_dict_like_saved_int8_supports_per_channel(self) -> None:
        mod = load_module()
        state = {
            "conv.weight": torch.tensor(
                [
                    [[[0.10, 0.20], [0.30, 0.40]]],
                    [[[1.00, 2.00], [3.00, 4.00]]],
                ],
                dtype=torch.float32,
            ),
            "conv.bias": torch.tensor([0.25, -0.5], dtype=torch.float32),
        }

        quantized = mod.quantize_state_dict_like_saved_int8(state, per_channel=True)

        self.assertEqual(tuple(quantized["conv.weight"].shape), (2, 1, 2, 2))
        self.assertFalse(torch.equal(quantized["conv.weight"][0], quantized["conv.weight"][1]))
        self.assertTrue(torch.equal(quantized["conv.bias"], state["conv.bias"]))

    def test_autocast_context_is_noop_on_cpu(self) -> None:
        mod = load_module()
        ctx = mod.autocast_context(torch.device("cpu"), enabled=True)
        self.assertIsInstance(ctx, nullcontext)

    def test_average_state_updates_running_mean(self) -> None:
        mod = load_module()
        base = {"w": torch.tensor([1.0, 3.0]), "n": torch.tensor([1], dtype=torch.int64)}
        avg = mod.init_average_state(base)
        mod.update_average_state(avg, {"w": torch.tensor([3.0, 5.0]), "n": torch.tensor([2], dtype=torch.int64)}, count=1)
        self.assertTrue(torch.allclose(avg["w"], torch.tensor([2.0, 4.0])))
        self.assertEqual(int(avg["n"][0]), 2)

    def test_parser_accepts_restart_and_swa_flags(self) -> None:
        mod = load_module()
        args = mod.build_arg_parser().parse_args(
            ["--restart-t0", "20", "--restart-tmult", "3", "--restart-eta-min", "1e-5", "--swa-start-epoch", "200", "--swa-every", "5"]
        )
        self.assertEqual(args.restart_t0, 20)
        self.assertEqual(args.restart_tmult, 3)
        self.assertEqual(args.swa_start_epoch, 200)
        self.assertEqual(args.swa_every, 5)

    def test_parser_accepts_quantized_checkpoint_selection_flags(self) -> None:
        mod = load_module()
        args = mod.build_arg_parser().parse_args(
            ["--checkpoint-select-int8", "--per-channel-int8"]
        )
        self.assertTrue(args.checkpoint_select_int8)
        self.assertTrue(args.per_channel_int8)


if __name__ == "__main__":
    unittest.main()
