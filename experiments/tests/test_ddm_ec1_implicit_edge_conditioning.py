from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments import ddm_ec1_implicit_edge_conditioning as ec1
from experiments.ddm_ec1_runtime import ec1_latent_conditioner as receiver
from tac.payload_retention_gate import check_no_measure_and_discard_payload


class _IdentityBlock(nn.Module):
    def forward(self, value: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        del frame
        return value


class _TinySemantic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(5, 96)
        self.coord_mix = nn.Conv2d(100, 96, 1)
        self.frame_embed = nn.Embedding(8, 8)
        self.blocks = nn.ModuleList([_IdentityBlock()])
        self.head = nn.Conv2d(96, 3, 3, padding=1)

    @staticmethod
    def coordinates(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 4, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, 5, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((xx, yy, xx.square(), yy.square()))[None].expand(batch, -1, -1, -1)

    def forward(self, tokens: torch.Tensor, pair_indices: torch.Tensor) -> torch.Tensor:
        value = self.token_embed(tokens).permute(0, 3, 1, 2)
        value = self.coord_mix(torch.cat((value, self.coordinates(len(tokens), value.device, value.dtype)), dim=1))
        frame = self.frame_embed(pair_indices)
        for block in self.blocks:
            value = block(value, frame)
        return torch.sigmoid(self.head(torch.nn.functional.gelu(value))) * 255.0


def test_context_is_receiver_computable_and_families_are_distinct() -> None:
    tokens = torch.tensor([[[0, 0, 1], [0, 2, 1]]])
    class_only = receiver.edge_context(tokens, "class_only")
    undirected = receiver.edge_context(tokens, "undirected")
    oriented = receiver.edge_context(tokens, "oriented")
    assert class_only.shape == (1, receiver.CONTEXT_CHANNELS, 2, 3)
    assert torch.count_nonzero(class_only[:, 5:]) == 0
    assert torch.count_nonzero(undirected[:, 5:10]) > 0
    assert torch.count_nonzero(undirected[:, 10:]) == 0
    assert torch.count_nonzero(oriented[:, 10:]) > 0
    assert not torch.equal(undirected, oriented)


def test_numpy_oriented_code_preserves_direction_beyond_undirected_code() -> None:
    left_edge = np.array([[[0, 1, 1], [0, 0, 0]]], dtype=np.uint8)
    right_edge = left_edge[:, :, ::-1].copy()
    undirected_left = ec1.context_codes(left_edge, "undirected")
    undirected_right = ec1.context_codes(right_edge, "undirected")
    oriented_left = ec1.context_codes(left_edge, "oriented")
    oriented_right = ec1.context_codes(right_edge, "oriented")
    assert np.array_equal(np.sort(np.unique(undirected_left)), np.sort(np.unique(undirected_right)))
    assert not np.array_equal(oriented_left, oriented_right)


def test_zero_adapter_is_exact_identity_and_nonzero_adapter_changes_latent_path(tmp_path: Path) -> None:
    torch.manual_seed(4)
    semantic = _TinySemantic().eval()
    tokens = torch.tensor(
        [[[0, 0, 1, 1, 1], [0, 2, 2, 1, 1], [0, 2, 3, 3, 1], [0, 0, 3, 1, 1]]]
    )
    indices = torch.tensor([3])
    identity = ec1.serialize_module(
        ec1.build_model(torch, "oriented", identity=True), tmp_path, "identity"
    )
    nonzero = ec1.serialize_module(
        ec1.build_model(torch, "oriented", identity=False), tmp_path, "nonzero"
    )
    with torch.inference_mode():
        base = semantic(tokens, indices)
        observed_identity = receiver.conditioned_semantic_forward(
            semantic, tokens, indices, Path(identity["coded"]["path"]).read_bytes()
        )
        observed_nonzero = receiver.conditioned_semantic_forward(
            semantic, tokens, indices, Path(nonzero["coded"]["path"]).read_bytes()
        )
    torch.testing.assert_close(observed_identity, base, rtol=0.0, atol=0.0)
    assert not torch.equal(observed_nonzero, base)


def test_counted_archive_is_deterministic_and_preserves_base_member(tmp_path: Path) -> None:
    base = tmp_path / "base.zip"
    with zipfile.ZipFile(base, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("p", b"cp135 payload")
    first = ec1.deterministic_archive(base, b"trained latent adapter")
    second = ec1.deterministic_archive(base, b"trained latent adapter")
    assert first == second
    candidate = tmp_path / "candidate.zip"
    candidate.write_bytes(first)
    with zipfile.ZipFile(candidate) as archive:
        assert archive.namelist() == ["p", "ec1_latent.br"]
        assert archive.read("p") == b"cp135 payload"
        assert archive.read("ec1_latent.br") == b"trained latent adapter"


def test_module_codec_parseback_and_repeat_are_exact(tmp_path: Path) -> None:
    exported = ec1.serialize_module(
        ec1.build_model(torch, "oriented", identity=False), tmp_path, "module"
    )
    coded = Path(exported["coded"]["path"]).read_bytes()
    repeated = Path(exported["repeat"]["path"]).read_bytes()
    header, arrays = receiver.parse_module(coded)
    assert coded == repeated
    assert header["family"] == "oriented"
    assert set(arrays) == {
        "context.bias",
        "context.weight",
        "depthwise.bias",
        "depthwise.weight",
        "head.bias",
        "head.weight",
    }


def test_ec1_sources_pass_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=Path.cwd(),
        strict=False,
        roots=(
            "experiments/ddm_ec1_implicit_edge_conditioning.py",
            "experiments/ddm_ec1_runtime/ec1_latent_conditioner.py",
            "experiments/tests/test_ddm_ec1_implicit_edge_conditioning.py",
        ),
    )
    assert findings == []
