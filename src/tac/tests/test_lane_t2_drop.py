from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from tac.experiments import train_renderer


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_t2drop_bootstrap.sh"


class ToyMaskModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(5, 4)
        with torch.no_grad():
            self.embedding.weight.fill_(1.0)

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        return self.embedding(masks)


def test_no_mask_encoder_flag_exists_in_argparse() -> None:
    args = train_renderer.parse_args([
        "--tag",
        "t2_drop_test",
        "--no-mask-encoder",
        "--no-auth-eval-on-best",
    ])

    assert args.no_mask_encoder is True


def test_with_flag_encoder_output_is_zeros_of_expected_shape() -> None:
    model = ToyMaskModel()
    args = train_renderer.parse_args([
        "--tag",
        "t2_drop_test",
        "--no-mask-encoder",
        "--no-auth-eval-on-best",
    ])
    handles = train_renderer.apply_no_mask_encoder_if_requested(model, args)

    out = model(torch.tensor([[0, 1, 2], [3, 4, 0]], dtype=torch.long))

    assert handles
    assert out.shape == (2, 3, 4)
    assert torch.count_nonzero(out).item() == 0
    for handle in handles:
        handle.remove()


def test_without_flag_encoder_output_is_nonzero() -> None:
    model = ToyMaskModel()
    args = train_renderer.parse_args([
        "--tag",
        "t2_drop_test",
        "--no-auth-eval-on-best",
    ])
    handles = train_renderer.apply_no_mask_encoder_if_requested(model, args)

    out = model(torch.tensor([[0, 1, 2]], dtype=torch.long))

    assert handles == []
    assert torch.count_nonzero(out).item() == out.numel()


def test_deploy_script_sets_informational_only_provenance_tag() -> None:
    text = SCRIPT.read_text()

    assert "informational_only" in text
    assert "true" in text.lower()
