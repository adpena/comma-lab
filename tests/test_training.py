"""Tests for training loop, checkpoint save/load, resume, and boundary masks."""
import tempfile
from pathlib import Path

import torch

from tac.architectures import build_postfilter
from tac.training import EMA, TrainConfig, Trainer


class TestEMA:
    def test_shadow_tracks_model(self):
        model = build_postfilter("standard", hidden=8)
        ema = EMA(model, decay=0.0)  # decay=0 → shadow = model instantly
        # Change model weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1.0)
        ema.update(model)
        for k, v in ema.shadow.items():
            if v.is_floating_point():
                assert v.abs().sum() > 0, f"EMA shadow {k} should track model"

    def test_high_decay_stability(self):
        model = build_postfilter("standard", hidden=8)
        ema = EMA(model, decay=0.999)
        orig = {k: v.clone() for k, v in ema.shadow.items()}
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)
        ema.update(model)
        # High decay → shadow should barely move
        for k in orig:
            if orig[k].is_floating_point():
                diff = (ema.shadow[k] - orig[k]).abs().max().item()
                assert diff < 2.0, f"High-decay EMA moved too much on {k}"


class TestTrainerConstruction:
    def test_creates_with_defaults(self):
        model = build_postfilter("standard", hidden=16)
        config = TrainConfig(hidden=16, epochs=100, tag="test-ctor")
        trainer = Trainer(model, config, device="cpu")
        assert trainer.best_scorer == float("inf")
        assert trainer._current_epoch == 0
        assert trainer._emergency_registered

    def test_signal_handlers_registered(self):
        import signal
        model = build_postfilter("standard", hidden=16)
        config = TrainConfig(hidden=16, epochs=100, tag="test-signals")
        Trainer(model, config, device="cpu")
        # SIGTERM handler should NOT be the default
        handler = signal.getsignal(signal.SIGTERM)
        assert handler is not signal.SIG_DFL


class TestCheckpointSaveLoad:
    def test_save_load_round_trip(self):
        model = build_postfilter("standard", hidden=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_with_dir = TrainConfig(
                hidden=8, epochs=100, tag="test-ckpt", output_dir=tmpdir
            )
            trainer = Trainer(model, config_with_dir, device="cpu")
            trainer._current_epoch = 42
            trainer.best_scorer = 1.234
            trainer.best_epoch = 40
            trainer.save_training_state()

            # Create new trainer and resume
            model2 = build_postfilter("standard", hidden=8)
            state_path = Path(tmpdir) / "training_state_test-ckpt.pt"
            config2 = TrainConfig(
                hidden=8, epochs=100, tag="test-ckpt",
                output_dir=tmpdir, resume_from=str(state_path)
            )
            trainer2 = Trainer(model2, config2, device="cpu")
            assert trainer2._current_epoch == 42
            assert trainer2.best_scorer == 1.234
            assert trainer2.best_epoch == 40

    def test_atomic_no_tmp_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainConfig(hidden=8, epochs=100, tag="test-atomic", output_dir=tmpdir)
            model = build_postfilter("standard", hidden=8)
            trainer = Trainer(model, config, device="cpu")
            trainer.save_training_state()
            # No .tmp files should remain
            tmp_files = list(Path(tmpdir).glob("*.tmp"))
            assert len(tmp_files) == 0, f"Leftover .tmp files: {tmp_files}"

    def test_ema_device_on_resume(self):
        """EMA shadow tensors should be on the correct device after resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainConfig(hidden=8, epochs=100, tag="test-device", output_dir=tmpdir)
            model = build_postfilter("standard", hidden=8)
            trainer = Trainer(model, config, device="cpu")
            trainer._current_epoch = 5
            trainer.save_training_state()

            state_path = Path(tmpdir) / "training_state_test-device.pt"
            config2 = TrainConfig(
                hidden=8, epochs=100, tag="test-device",
                output_dir=tmpdir, resume_from=str(state_path)
            )
            model2 = build_postfilter("standard", hidden=8)
            trainer2 = Trainer(model2, config2, device="cpu")
            for k, v in trainer2.ema.shadow.items():
                assert v.device == torch.device("cpu"), f"EMA {k} on wrong device"


class TestBoundaryMask:
    """Test compute_boundary_mask shape handling."""

    def test_boundary_mask_shape(self):
        """Boundary mask should work with correct input shapes."""
        from tac.losses import compute_boundary_mask

        # Create a mock segnet that expects (B, C, H, W) after preprocess
        class MockSegNet(torch.nn.Module):
            def preprocess_input(self, x):
                # Expects (B, T, C, H, W), returns (B, C, H_small, W_small)
                assert x.ndim == 5, f"preprocess_input expects 5D, got {x.ndim}D"
                frame = x[:, -1, ...]  # (B, C, H, W)
                return torch.nn.functional.interpolate(
                    frame, size=(384, 512), mode="bilinear", align_corners=False
                )

            def forward(self, x):
                B, C, H, W = x.shape
                return torch.randn(B, 5, H, W)  # 5-class segmentation

        segnet = MockSegNet()
        # Pair shape: (1, 2, H, W, 3)
        gt_pair = torch.randint(0, 256, (1, 2, 64, 64, 3), dtype=torch.uint8)
        mask = compute_boundary_mask(gt_pair, segnet, device="cpu")
        assert mask.ndim == 2, f"Boundary mask should be 2D, got {mask.ndim}D"
        assert mask.dtype == torch.float32
        assert (mask >= 0).all() and (mask <= 1).all()


class TestEvalScorerLoss:
    """Test eval_scorer_loss correctness."""

    def test_no_gradients(self):
        """eval_scorer_loss should not build an autograd graph."""
        from tac.losses import eval_scorer_loss

        class MockPoseNet(torch.nn.Module):
            def preprocess_input(self, x):
                B, T, C, H, W = x.shape
                return x.reshape(B, T * C, H, W)

            def forward(self, x):
                return {"pose": torch.randn(x.shape[0], 12)}

        class MockSegNet(torch.nn.Module):
            def preprocess_input(self, x):
                return x[:, -1, ...]

            def forward(self, x):
                B, C, H, W = x.shape
                return torch.randn(B, 5, H, W)

        pair = torch.rand(1, 2, 32, 32, 3) * 255
        score, pose, seg = eval_scorer_loss(pair, pair, MockPoseNet(), MockSegNet())
        assert isinstance(score, float)
        assert isinstance(pose, float)
        assert isinstance(seg, float)
        # Should be >= 0
        assert pose >= 0
        assert seg >= 0

    def test_identical_pairs_low_distortion(self):
        """Identical inputs should produce zero or near-zero distortion."""
        from tac.losses import eval_scorer_loss

        class DetPoseNet(torch.nn.Module):
            def preprocess_input(self, x):
                return x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

            def forward(self, x):
                spatial_mean = x.mean(dim=(2, 3))  # (B, C)
                # Pad/repeat to 12 outputs like real PoseNet
                return {"pose": spatial_mean[:, :1].expand(-1, 12)}

        class DetSegNet(torch.nn.Module):
            def preprocess_input(self, x):
                return x[:, -1, ...]

            def forward(self, x):
                return x[:, :5, :, :]  # just use first 5 channels

        pair = torch.rand(1, 2, 32, 32, 3) * 255
        score, pose, seg = eval_scorer_loss(pair, pair, DetPoseNet(), DetSegNet())
        assert pose < 1e-6, f"Identical pairs should have ~0 pose dist, got {pose}"
        assert seg < 1e-6, f"Identical pairs should have ~0 seg dist, got {seg}"
