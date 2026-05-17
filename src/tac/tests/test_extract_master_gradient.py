# SPDX-License-Identifier: MIT
"""Tests for tools/extract_master_gradient.py.

Validates the per-byte gradient projection on a tiny synthetic decoder so the
extractor logic is exercised without requiring the full ~178517-byte fec6
archive + GT video decode + scorer load. The synthetic decoder mirrors the
fec6 codec's int8 mantissa + fp16 scale grammar at small scale.

[verified-against: .omx/research/grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517.md §3.2]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
TOOLS_DIR = REPO_ROOT / "tools"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# Import lazily — `extract_master_gradient` imports torch + numpy at module-load,
# fine; but it also imports tac.differentiable_eval_roundtrip + tac.scorer.
import extract_master_gradient as emg  # noqa: E402

from tac.master_gradient import (  # noqa: E402
    AGGREGATE_GRADIENT_TENSOR_KIND,
    CONTEST_RATE_DENOM_BYTES,
    PER_PAIR_GRADIENT_TENSOR_KIND,
    MasterGradient,
    OperatingPoint,
    compute_marginal_coefficients,
    load_anchors_lenient,
    update_from_anchor,
)

# ─────────────────────────────────────────────────────────────────────────── #
# Synthetic small decoder + matching codec module                             #
# ─────────────────────────────────────────────────────────────────────────── #


class _TinyDecoder(nn.Module):
    """3-param decoder for unit tests; mirrors HNeRVDecoder API surface."""

    def __init__(self, latent_dim=4, base_channels=4, eval_size=(2, 2)):
        super().__init__()
        self.eval_size = eval_size
        self.stem = nn.Linear(latent_dim, base_channels * eval_size[0] * eval_size[1])
        # Final maps to a 2-frame, 3-channel pair (B, 2, 3, H, W)
        self.head_0 = nn.Linear(base_channels * eval_size[0] * eval_size[1], 3 * eval_size[0] * eval_size[1])
        self.head_1 = nn.Linear(base_channels * eval_size[0] * eval_size[1], 3 * eval_size[0] * eval_size[1])

    def forward(self, z):
        B = z.shape[0]
        h = self.stem(z)  # (B, C*H*W)
        h = torch.relu(h)
        f0 = torch.sigmoid(self.head_0(h)).view(B, 3, *self.eval_size) * 255.0
        f1 = torch.sigmoid(self.head_1(h)).view(B, 3, *self.eval_size) * 255.0
        return torch.stack([f0, f1], dim=1)  # (B, 2, 3, H, W)


# ─────────────────────────────────────────────────────────────────────────── #
# Test: per-tensor Jacobian scale projection                                  #
# ─────────────────────────────────────────────────────────────────────────── #


def test_project_per_param_gradient_to_per_byte_uniform_spread(tmp_path):
    """A 1-tensor synthetic layout: per-byte sensitivity sums to per-tensor sensitivity."""
    # Construct a synthetic layout: 1 tensor of 4 weights, fp16 scale 2.0
    span = emg._TensorByteSpan(
        name="stem.weight",
        storage_index=0,
        shape=(4,),
        numel=4,
        mantissa_byte_offset=0,
        scale_byte_offset=4,
        fp16_scale=2.0,
        byte_map="twos",
    )
    layout = emg._Fec6ArchiveLayout(
        archive_path=tmp_path / "fake.bin",
        archive_sha256="a" * 64,
        archive_bytes=b"\x00" * 100,
        n_archive_bytes=100,
        decoder_blob_offset=0,
        decoder_blob_len=20,  # synthetic compressed-region size
        decoder_tensor_spans=(span,),
        decoder_raw_decompressed=b"\x00" * 6,  # 4 mantissa + 2 scale
        latent_blob_offset=20,
        latent_blob_len=40,
        sidecar_blob_offset=60,
        sidecar_blob_len=40,
        n_pairs=10,
        latent_dim=4,
        base_channels=4,
        eval_size=(2, 2),
        has_fp11_outer_wrapper=False,
    )

    # Per-weight gradient: [1.0, 2.0, 3.0, 4.0]; expected per-tensor mass = (1+2+3+4) * |scale|
    grad_seg = {"stem.weight": torch.tensor([1.0, 2.0, 3.0, 4.0])}
    grad_pose = {"stem.weight": torch.tensor([0.5, 0.5, 0.5, 0.5])}

    G = emg.project_per_param_gradient_to_per_byte(layout, grad_seg, grad_pose)

    assert G.shape == (100, 3)
    # Mass conservation in decoder region [0, 20): sum of seg col == sum of |per_byte_grad * scale|
    expected_seg_mass = sum(abs(v) for v in [1.0, 2.0, 3.0, 4.0]) * 2.0
    expected_pose_mass = 4 * 0.5 * 2.0
    actual_seg_mass = G[0:20, 0].sum()
    actual_pose_mass = G[0:20, 1].sum()
    np.testing.assert_allclose(actual_seg_mass, expected_seg_mass, rtol=1e-3)
    np.testing.assert_allclose(actual_pose_mass, expected_pose_mass, rtol=1e-3)

    # Rate column uniform across all 100 bytes
    expected_rate = 1.0 / CONTEST_RATE_DENOM_BYTES
    np.testing.assert_allclose(G[:, 2], expected_rate, rtol=1e-6)


def test_project_zero_gradient_yields_zero_seg_and_pose_columns(tmp_path):
    """If the per-param grad is all zero, the per-byte seg/pose columns must be zero too."""
    span = emg._TensorByteSpan(
        name="stem.weight",
        storage_index=0,
        shape=(2,),
        numel=2,
        mantissa_byte_offset=0,
        scale_byte_offset=2,
        fp16_scale=1.0,
        byte_map="twos",
    )
    layout = emg._Fec6ArchiveLayout(
        archive_path=tmp_path / "fake.bin",
        archive_sha256="b" * 64,
        archive_bytes=b"\x00" * 50,
        n_archive_bytes=50,
        decoder_blob_offset=0,
        decoder_blob_len=10,
        decoder_tensor_spans=(span,),
        decoder_raw_decompressed=b"\x00" * 4,
        latent_blob_offset=10,
        latent_blob_len=20,
        sidecar_blob_offset=30,
        sidecar_blob_len=20,
        n_pairs=2,
        latent_dim=2,
        base_channels=1,
        eval_size=(1, 1),
        has_fp11_outer_wrapper=False,
    )
    grad_seg = {"stem.weight": torch.zeros(2)}
    grad_pose = {"stem.weight": torch.zeros(2)}

    G = emg.project_per_param_gradient_to_per_byte(layout, grad_seg, grad_pose)
    assert G.shape == (50, 3)
    assert np.allclose(G[:, 0], 0.0)
    assert np.allclose(G[:, 1], 0.0)
    # Rate column still uniform (non-zero)
    assert (G[:, 2] > 0).all()


def test_project_handles_missing_tensor_in_grad_dict(tmp_path):
    """If a span has no entry in grad_seg/grad_pose, it must be skipped (not raise)."""
    span = emg._TensorByteSpan(
        name="never.seen",
        storage_index=0,
        shape=(1,),
        numel=1,
        mantissa_byte_offset=0,
        scale_byte_offset=1,
        fp16_scale=1.0,
        byte_map="twos",
    )
    layout = emg._Fec6ArchiveLayout(
        archive_path=tmp_path / "fake.bin",
        archive_sha256="c" * 64,
        archive_bytes=b"",
        n_archive_bytes=10,
        decoder_blob_offset=0,
        decoder_blob_len=5,
        decoder_tensor_spans=(span,),
        decoder_raw_decompressed=b"\x00\x00\x00",
        latent_blob_offset=5,
        latent_blob_len=3,
        sidecar_blob_offset=8,
        sidecar_blob_len=2,
        n_pairs=1,
        latent_dim=1,
        base_channels=1,
        eval_size=(1, 1),
        has_fp11_outer_wrapper=False,
    )

    G = emg.project_per_param_gradient_to_per_byte(layout, {}, {})
    assert G.shape == (10, 3)
    assert np.allclose(G[:, 0], 0.0)
    assert np.allclose(G[:, 1], 0.0)


# ─────────────────────────────────────────────────────────────────────────── #
# Test: forward + backward autograd reaches decoder weights                   #
# ─────────────────────────────────────────────────────────────────────────── #


def test_synthetic_decoder_backward_yields_nonzero_grads():
    """Confirm that backward of a quadratic-style loss reaches synthetic decoder weights."""
    decoder = _TinyDecoder(latent_dim=4, base_channels=4, eval_size=(2, 2))
    z = torch.randn(2, 4)
    decoded = decoder(z)  # (2, 2, 3, 2, 2)
    # Quadratic loss for fast-and-correct gradient check
    loss = (decoded - 100.0).pow(2).mean()
    loss.backward()
    for name, p in decoder.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert (p.grad.abs().sum() > 0).item(), f"{name} grad sum is 0"


# ─────────────────────────────────────────────────────────────────────────── #
# Test: anchor persistence roundtrip (uses canonical helper)                   #
# ─────────────────────────────────────────────────────────────────────────── #


def test_append_anchor_locked_roundtrip(tmp_path):
    """Write + read back a MasterGradient anchor via canonical helper."""
    from tac.master_gradient import append_anchor_locked

    sidecar = tmp_path / "grad.npy"
    np.save(sidecar, np.zeros((10, 3), dtype=np.float32))

    op = OperatingPoint(d_seg=0.05, d_pose=1e-5, rate=0.119, score=0.193)
    grad = MasterGradient(
        archive_sha256="a" * 64,
        operating_point=op,
        gradient_array_path=str(sidecar),
        n_bytes=10,
        measurement_method="autograd_per_parameter_projected_fec6_int8_fp16_jacobian",
        measurement_axis="[contest-CPU]",
        measurement_hardware="linux_x86_64_modal_cpu",
        measurement_call_id=None,
        measurement_utc="2026-05-17T12:00:00+00:00",
    )

    ledger = tmp_path / "anchors.jsonl"
    lock = tmp_path / ".lock"
    append_anchor_locked(grad, path=ledger, lock_path=lock)
    rows = load_anchors_lenient(ledger)
    assert len(rows) == 1
    row = rows[0]
    assert row["archive_sha256"] == "a" * 64
    assert row["measurement_axis"] == "[contest-CPU]"
    assert row["measurement_hardware"] == "linux_x86_64_modal_cpu"
    assert row["operating_point"]["d_seg"] == 0.05
    assert row["schema_version"] == "master_gradient_anchor_v1"
    assert row["gradient_tensor_kind"] == AGGREGATE_GRADIENT_TENSOR_KIND


def test_per_pair_master_gradient_load_and_predict_vector(tmp_path):
    """Per-pair anchors are typed and preserve the pair axis instead of collapsing it."""
    sidecar = tmp_path / "grad_per_pair.npy"
    arr = np.zeros((4, 3, 3), dtype=np.float32)
    arr[1, :, 0] = np.array([0.01, -0.02, 0.03], dtype=np.float32)
    arr[1, :, 1] = np.array([0.001, 0.002, -0.003], dtype=np.float32)
    np.save(sidecar, arr)

    op = OperatingPoint(d_seg=0.05, d_pose=1e-5, rate=0.119, score=0.193)
    grad = MasterGradient(
        archive_sha256="b" * 64,
        operating_point=op,
        gradient_array_path=str(sidecar),
        n_bytes=4,
        measurement_method="autograd_per_parameter_projected_fec6_int8_fp16_jacobian_per_pair_3pair",
        measurement_axis="[contest-CPU]",
        measurement_hardware="linux_x86_64_modal_cpu",
        measurement_call_id="call_per_pair",
        measurement_utc="2026-05-17T12:00:00+00:00",
        gradient_tensor_kind=PER_PAIR_GRADIENT_TENSOR_KIND,
        n_pairs=3,
    )

    loaded = grad.load_per_pair_gradient()
    assert loaded.shape == (4, 3, 3)
    delta = grad.predict_delta_s_per_pair({1: 2.0})
    assert delta.shape == (3,)
    assert delta[0] != delta[1]
    assert delta[1] != delta[2]

    with pytest.raises(ValueError, match="not an aggregate"):
        grad.load_gradient()


def test_per_pair_anchor_requires_n_pairs(tmp_path):
    sidecar = tmp_path / "grad_per_pair.npy"
    np.save(sidecar, np.zeros((4, 3, 3), dtype=np.float32))
    op = OperatingPoint(d_seg=0.05, d_pose=1e-5, rate=0.119, score=0.193)

    with pytest.raises(ValueError, match="positive integer n_pairs"):
        MasterGradient(
            archive_sha256="c" * 64,
            operating_point=op,
            gradient_array_path=str(sidecar),
            n_bytes=4,
            measurement_method="autograd_per_parameter_projected_fec6_int8_fp16_jacobian_per_pair_3pair",
            measurement_axis="[contest-CPU]",
            measurement_hardware="linux_x86_64_modal_cpu",
            measurement_call_id=None,
            measurement_utc="2026-05-17T12:00:00+00:00",
            gradient_tensor_kind=PER_PAIR_GRADIENT_TENSOR_KIND,
        )


def test_update_from_anchor_preserves_per_pair_metadata(tmp_path):
    sidecar = tmp_path / "grad_per_pair.npy"
    np.save(sidecar, np.zeros((5, 2, 3), dtype=np.float32))
    ledger = tmp_path / "anchors.jsonl"

    update_from_anchor(
        {
            "archive_sha256": "d" * 64,
            "operating_point": {
                "d_seg": 0.05,
                "d_pose": 1e-5,
                "rate": 0.119,
                "score": 0.193,
            },
            "gradient_array_path": str(sidecar),
            "n_bytes": 5,
            "measurement_method": "autograd_per_parameter_projected_fec6_int8_fp16_jacobian_per_pair_2pair",
            "measurement_axis": "[contest-CPU]",
            "measurement_hardware": "linux_x86_64_modal_cpu",
            "measurement_call_id": "call_pp",
            "measurement_utc": "2026-05-17T12:00:00+00:00",
            "gradient_tensor_kind": PER_PAIR_GRADIENT_TENSOR_KIND,
            "n_pairs": 2,
        },
        path=ledger,
    )

    rows = load_anchors_lenient(ledger)
    assert rows[0]["gradient_tensor_kind"] == PER_PAIR_GRADIENT_TENSOR_KIND
    assert rows[0]["n_pairs"] == 2


# ─────────────────────────────────────────────────────────────────────────── #
# Test: rate-column analytical correctness                                     #
# ─────────────────────────────────────────────────────────────────────────── #


def test_rate_column_uses_canonical_denom(tmp_path):
    """Rate column must equal 1.0 / CONTEST_RATE_DENOM_BYTES regardless of operating point."""
    layout = emg._Fec6ArchiveLayout(
        archive_path=tmp_path / "fake.bin",
        archive_sha256="d" * 64,
        archive_bytes=b"\x00" * 5,
        n_archive_bytes=5,
        decoder_blob_offset=0,
        decoder_blob_len=2,
        decoder_tensor_spans=(),
        decoder_raw_decompressed=b"",
        latent_blob_offset=2,
        latent_blob_len=2,
        sidecar_blob_offset=4,
        sidecar_blob_len=1,
        n_pairs=1,
        latent_dim=1,
        base_channels=1,
        eval_size=(1, 1),
        has_fp11_outer_wrapper=False,
    )
    G = emg.project_per_param_gradient_to_per_byte(layout, {}, {})
    expected = 1.0 / CONTEST_RATE_DENOM_BYTES
    # rtol=1e-6 accounts for float32-vs-float64 representation of the canonical denom
    np.testing.assert_allclose(G[:, 2], expected, rtol=1e-6)
    # Per CLAUDE.md "Apples-to-apples evidence discipline": the marginal application is
    # downstream (predict_delta_s multiplies the rate column by the 25.0 marginal).
    op = OperatingPoint(d_seg=0.05, d_pose=1e-5, rate=0.119, score=0.193)
    _, _, rate_marg = compute_marginal_coefficients(op)
    # rate_marg = 25 / 37545489; rate_col * rate_marg gives ΔS per byte-delta from rate
    np.testing.assert_allclose(rate_marg, 25.0 / CONTEST_RATE_DENOM_BYTES, rtol=1e-12)


# ─────────────────────────────────────────────────────────────────────────── #
# Test: CLI smoke (extractor refuses /tmp output per Catalog #220)             #
# ─────────────────────────────────────────────────────────────────────────── #


def test_main_refuses_tmp_output_path(tmp_path, monkeypatch):
    """The CLI MUST refuse --output-npy under /tmp/ per CLAUDE.md transient-evidence trap."""
    fake_archive = tmp_path / "fake.zip"
    fake_archive.write_bytes(b"\x00" * 10)
    fake_inflate = tmp_path / "inflate.py"
    fake_inflate.write_text("# noop")
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    fake_upstream = tmp_path / "upstream"
    fake_upstream.mkdir(parents=True, exist_ok=True)

    argv = [
        "--archive", str(fake_archive),
        "--inflate-py", str(fake_inflate),
        "--upstream-dir", str(fake_upstream),
        "--axis", "[contest-CPU]",
        "--output-npy", "/tmp/forbidden.npy",
        "--device", "cpu",
    ]
    with pytest.raises(SystemExit) as excinfo:
        emg.main(argv)
    assert "Forbidden /tmp paths" in str(excinfo.value) or "Catalog #220" in str(excinfo.value)


# ─────────────────────────────────────────────────────────────────────────── #
# Test: archive-zip extraction (synthetic 1-member zip)                        #
# ─────────────────────────────────────────────────────────────────────────── #


def test_maybe_extract_inner_archive_from_zip_single_member(tmp_path):
    import zipfile
    z = tmp_path / "archive.zip"
    inner = b"hello world payload"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("0.bin", inner)
    out = emg._maybe_extract_inner_archive_from_zip(z)
    assert out == inner


def test_maybe_extract_inner_archive_from_zip_canonical_name(tmp_path):
    import zipfile
    z = tmp_path / "archive.zip"
    inner = b"the canonical archive"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("0.bin", inner)
        zf.writestr("metadata.json", b"{}")
    out = emg._maybe_extract_inner_archive_from_zip(z)
    assert out == inner


def test_maybe_extract_inner_archive_from_zip_raw_bytes_passthrough(tmp_path):
    raw = tmp_path / "raw.bin"
    raw.write_bytes(b"\x00\x01\x02not_a_zip")
    out = emg._maybe_extract_inner_archive_from_zip(raw)
    assert out == b"\x00\x01\x02not_a_zip"


# ─────────────────────────────────────────────────────────────────────────── #
# Test: operating point validation                                              #
# ─────────────────────────────────────────────────────────────────────────── #


def test_operating_point_rejects_zero_d_pose():
    """d_pose must be > 0 (marginal 5/sqrt(10*d_pose) is undefined at 0)."""
    with pytest.raises(ValueError, match="d_pose must be > 0"):
        OperatingPoint(d_seg=0.05, d_pose=0.0, rate=0.119, score=0.193)


def test_marginal_coefficients_at_fec6_operating_point():
    """At fec6's known operating point (d_seg ~0.05, d_pose ~1e-5), marginals are computable."""
    op = OperatingPoint(d_seg=0.05, d_pose=1e-5, rate=0.119, score=0.193)
    seg, pose, rate = compute_marginal_coefficients(op)
    assert seg == 100.0
    # pose = 5 / sqrt(10 * 1e-5) = 5 / sqrt(1e-4) = 5 / 0.01 = 500
    assert abs(pose - 500.0) < 1e-3
    assert abs(rate - 25.0 / CONTEST_RATE_DENOM_BYTES) < 1e-12
