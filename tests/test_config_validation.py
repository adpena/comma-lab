"""Tests for TrainConfig pydantic validation."""
import pytest
from pydantic import ValidationError

from tac.training import TrainConfig


class TestTrainConfigValid:
    def test_defaults(self):
        c = TrainConfig()
        assert c.hidden == 64
        assert c.kernel == 3
        assert c.tag == "untitled"

    def test_custom(self):
        c = TrainConfig(hidden=96, epochs=2500, tag="h96_long")
        assert c.hidden == 96
        assert c.epochs == 2500

    def test_immutable(self):
        c = TrainConfig()
        with pytest.raises(ValidationError):
            c.hidden = 128  # type: ignore[misc]


class TestTrainConfigInvalid:
    def test_even_kernel(self):
        with pytest.raises(ValidationError, match="kernel must be odd"):
            TrainConfig(kernel=4)

    def test_hidden_too_small(self):
        with pytest.raises(ValidationError):
            TrainConfig(hidden=2)

    def test_hidden_too_large(self):
        with pytest.raises(ValidationError):
            TrainConfig(hidden=1024)

    def test_warmup_exceeds_epochs(self):
        with pytest.raises(ValidationError, match="warmup_epochs"):
            TrainConfig(epochs=10, warmup_epochs=10)

    def test_lr_zero(self):
        with pytest.raises(ValidationError):
            TrainConfig(lr=0.0)

    def test_lr_too_high(self):
        with pytest.raises(ValidationError):
            TrainConfig(lr=2.0)

    def test_bad_loss_mode(self):
        with pytest.raises(ValidationError):
            TrainConfig(loss_mode="invalid")

    def test_bad_scheduler(self):
        with pytest.raises(ValidationError):
            TrainConfig(scheduler="step")

    def test_tag_with_spaces(self):
        with pytest.raises(ValidationError):
            TrainConfig(tag="bad tag name")

    def test_tag_empty(self):
        with pytest.raises(ValidationError):
            TrainConfig(tag="")

    def test_temperature_inverted(self):
        with pytest.raises(ValidationError, match="temperature_end"):
            TrainConfig(temperature_start=0.01, temperature_end=1.0)

    def test_negative_alpha(self):
        with pytest.raises(ValidationError):
            TrainConfig(alpha=-1.0)

    def test_ema_decay_bounds(self):
        with pytest.raises(ValidationError):
            TrainConfig(ema_decay=0.5)
        with pytest.raises(ValidationError):
            TrainConfig(ema_decay=1.0)

    def test_kl_distill_requires_high_temperature(self):
        with pytest.raises(ValidationError, match="kl_distill requires temperature_start"):
            TrainConfig(loss_mode="kl_distill", temperature_start=1.0, temperature_end=0.05)

    def test_kl_distill_requires_temperature_end_ge_01(self):
        with pytest.raises(ValidationError, match=r"temperature_end >= 0\.1"):
            TrainConfig(loss_mode="kl_distill", temperature_start=5.0, temperature_end=0.05)

    def test_kl_distill_allows_sub1_temperature(self):
        """T_end=0.2 is now allowed for argmax pressure phase."""
        c = TrainConfig(loss_mode="kl_distill", temperature_start=5.0, temperature_end=0.2)
        assert c.temperature_end == 0.2

    def test_kl_distill_valid_config(self):
        c = TrainConfig(loss_mode="kl_distill", temperature_start=5.0, temperature_end=1.0)
        assert c.loss_mode == "kl_distill"
        assert c.temperature_start == 5.0
