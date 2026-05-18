# SPDX-License-Identifier: MIT
"""Tests for tools/extract_master_gradient.py.

Validates the per-byte gradient projection on a tiny synthetic decoder so the
extractor logic is exercised without requiring the full ~178517-byte fec6
archive + GT video decode + scorer load. The synthetic decoder mirrors the
fec6 codec's int8 mantissa + fp16 scale grammar at small scale.

[verified-against: .omx/research/grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517.md §3.2]
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
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
# Synthetic archive grammar fixtures                                          #
# ─────────────────────────────────────────────────────────────────────────── #


def _zip_payload(payload: bytes, member_name: str = "x") -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(member_name, payload)
    return bio.getvalue()


def _a1_payload() -> bytes:
    section_total = 4 + 162_164
    return (
        section_total.to_bytes(4, "little")
        + b"\x1b\xcd\x03\xf8"
        + b"D" * (162_164 - 4)
        + b"L" * 15_387
        + b"SIDE"
    )


def _pr101_payload() -> bytes:
    return b"\x1b\xcd\x03\xf8" + b"D" * (162_164 - 4) + b"L" * 15_387 + b"SIDE"


def _pr106_payload() -> bytes:
    primary = b"P" * 5
    base_sidecar = b"B" * 511
    extra = b"EE"
    return (
        b"\xfe\x0d"
        + len(primary).to_bytes(4, "little")
        + primary
        + base_sidecar
        + len(extra).to_bytes(2, "little")
        + extra
        + b"FRAME!"
    )


def _pr106_ff_payload() -> bytes:
    decoder = b"D" * 7
    tail = b"T" * 11
    return b"\xff" + len(decoder).to_bytes(3, "little") + decoder + tail


def _hnerv_lc_v2_payload() -> bytes:
    parts = (b"D" * 7, b"S" * 56, b"L" * 9, b"W" * 3)
    return b"".join(len(part).to_bytes(4, "little") + part for part in parts)


def _pr107_payload() -> bytes:
    parts = (b"M" * 5, b"D" * 8, b"L" * 6)
    return b"".join(len(part).to_bytes(4, "little") + part for part in parts)


def _fec6_payload() -> bytes:
    source = _pr101_payload()
    selector = b"FEC6" + (600).to_bytes(2, "little") + b"SEL"
    return b"FP11" + len(source).to_bytes(4, "little") + source + len(selector).to_bytes(2, "little") + selector


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

    # Rate is a packet byte-count property, not a byte-value derivative.
    np.testing.assert_allclose(G[:, 2], 0.0, rtol=0, atol=0)


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
    # Rate column stays zero until a packet-valid candidate measures byte-count delta.
    assert np.allclose(G[:, 2], 0.0)


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


def test_master_gradient_records_scored_archive_and_subject_custody(tmp_path):
    """Charged ZIP custody and differentiated payload custody must not collapse."""
    sidecar = tmp_path / "grad.npy"
    np.save(sidecar, np.zeros((10, 3), dtype=np.float32))

    grad = MasterGradient(
        archive_sha256="a" * 64,
        scored_archive_sha256="a" * 64,
        scored_archive_bytes=110,
        gradient_subject_sha256="b" * 64,
        gradient_subject_bytes=10,
        gradient_byte_domain="zip_inner_member_payload",
        n_pairs_used=8,
        n_pairs_total=600,
        operating_point=OperatingPoint(
            d_seg=0.05, d_pose=1e-5, rate=110 / CONTEST_RATE_DENOM_BYTES, score=0.193
        ),
        gradient_array_path=str(sidecar),
        n_bytes=10,
        measurement_method="autograd_per_parameter_projected_fec6_int8_fp16_jacobian",
        measurement_axis="[diagnostic-CPU]",
        measurement_hardware="darwin_arm64_local_cpu_advisory",
        measurement_call_id=None,
        measurement_utc="2026-05-17T12:00:00+00:00",
    )

    assert grad.archive_sha256 == "a" * 64
    assert grad.scored_archive_bytes == 110
    assert grad.gradient_subject_sha256 == "b" * 64
    assert grad.gradient_subject_bytes == grad.n_bytes
    assert grad.gradient_byte_domain == "zip_inner_member_payload"


def test_master_gradient_refuses_subject_byte_count_mismatch(tmp_path):
    sidecar = tmp_path / "grad.npy"
    np.save(sidecar, np.zeros((10, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="gradient_subject_bytes must equal n_bytes"):
        MasterGradient(
            archive_sha256="a" * 64,
            gradient_subject_bytes=11,
            operating_point=OperatingPoint(d_seg=0.05, d_pose=1e-5, rate=0.119, score=0.193),
            gradient_array_path=str(sidecar),
            n_bytes=10,
            measurement_method="autograd_per_parameter_projected_fec6_int8_fp16_jacobian",
            measurement_axis="[diagnostic-CPU]",
            measurement_hardware="darwin_arm64_local_cpu_advisory",
            measurement_call_id=None,
            measurement_utc="2026-05-17T12:00:00+00:00",
        )


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


def test_update_from_anchor_preserves_archive_custody_metadata(tmp_path):
    sidecar = tmp_path / "grad.npy"
    np.save(sidecar, np.zeros((5, 3), dtype=np.float32))
    ledger = tmp_path / "anchors.jsonl"

    update_from_anchor(
        {
            "archive_sha256": "a" * 64,
            "scored_archive_sha256": "a" * 64,
            "scored_archive_bytes": 123,
            "gradient_subject_sha256": "b" * 64,
            "gradient_subject_bytes": 5,
            "gradient_byte_domain": "zip_inner_member_payload",
            "n_pairs_used": 8,
            "n_pairs_total": 600,
            "operating_point": {
                "d_seg": 0.05,
                "d_pose": 1e-5,
                "rate": 123 / CONTEST_RATE_DENOM_BYTES,
                "score": 0.193,
            },
            "gradient_array_path": str(sidecar),
            "n_bytes": 5,
            "measurement_method": "autograd_per_parameter_projected_fec6_int8_fp16_jacobian",
            "measurement_axis": "[diagnostic-CPU]",
            "measurement_hardware": "darwin_arm64_local_cpu_advisory",
            "measurement_call_id": "call_diag",
            "measurement_utc": "2026-05-17T12:00:00+00:00",
        },
        path=ledger,
    )

    row = load_anchors_lenient(ledger)[0]
    assert row["scored_archive_sha256"] == "a" * 64
    assert row["gradient_subject_sha256"] == "b" * 64
    assert row["gradient_byte_domain"] == "zip_inner_member_payload"
    assert row["n_pairs_used"] == 8
    assert row["n_pairs_total"] == 600


# ─────────────────────────────────────────────────────────────────────────── #
# Test: rate-column analytical correctness                                     #
# ─────────────────────────────────────────────────────────────────────────── #


def test_rate_column_is_zero_for_byte_value_derivatives(tmp_path):
    """Changing an existing byte value does not change archive byte count."""
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
    np.testing.assert_allclose(G[:, 2], 0.0, rtol=0, atol=0)
    # The byte-count marginal remains canonical; packet-valid candidates must
    # multiply it by a measured rate_bytes_delta after rebuilding the archive.
    op = OperatingPoint(d_seg=0.05, d_pose=1e-5, rate=0.119, score=0.193)
    _, _, rate_marg = compute_marginal_coefficients(op)
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


def test_validate_measurement_authority_refuses_subset_contest_axis():
    with pytest.raises(SystemExit, match="require the full pair set"):
        emg._validate_measurement_authority(
            axis="[contest-CPU]",
            device="cpu",
            hardware_substrate="linux_x86_64_modal_cpu",
            n_pairs_used=8,
            n_pairs_total=600,
        )


def test_validate_measurement_authority_allows_subset_diagnostic_axis():
    emg._validate_measurement_authority(
        axis="[diagnostic-CPU]",
        device="cpu",
        hardware_substrate="darwin_arm64_local_cpu_advisory",
        n_pairs_used=8,
        n_pairs_total=600,
    )


def test_validate_measurement_authority_refuses_advisory_contest_axis():
    with pytest.raises(SystemExit, match="cannot be written from advisory"):
        emg._validate_measurement_authority(
            axis="[contest-CPU]",
            device="cpu",
            hardware_substrate="darwin_arm64_local_cpu_advisory",
            n_pairs_used=600,
            n_pairs_total=600,
        )


# ─────────────────────────────────────────────────────────────────────────── #
# Test: archive grammar detection and typed boundaries                         #
# ─────────────────────────────────────────────────────────────────────────── #


@pytest.mark.parametrize(
    ("archive_bytes", "expected_grammar", "section_names", "projection_supported"),
    [
        (
            _zip_payload(_fec6_payload()),
            "fec6_fp11_selector",
            [
                "fp11_magic",
                "source_len_le_u32",
                "source_payload",
                "selector_len_le_u16",
                "selector_payload",
            ],
            True,
        ),
        (
            _zip_payload(_a1_payload()),
            "a1_finetuned",
            ["a1_section_header", "decoder", "latent", "sidecar"],
            False,
        ),
        (
            _zip_payload(_pr101_payload()),
            "pr101_lc_v2",
            ["decoder", "latent", "sidecar"],
            True,
        ),
        (
            _zip_payload(_pr106_payload()),
            "pr106_format0d",
            [
                "format_magic",
                "format_id",
                "pr106_len_le_u32",
                "pr106_payload",
                "base_format0c_sidecar_payload",
                "extra_payload_len_le_u16",
                "extra_pr101_ranked_no_op_payload",
                "extra_framing_meta",
            ],
            False,
        ),
        (
            _zip_payload(_pr106_ff_payload(), member_name="0.bin"),
            "pr106_ff_packed_hnerv",
            [
                "ff_magic",
                "decoder_len_u24le",
                "decoder_packed_brotli",
                "latents_and_sidecar_brotli",
            ],
            False,
        ),
        (
            _zip_payload(_hnerv_lc_v2_payload(), member_name="0.bin"),
            "hnerv_lc_v2_length_prefixed",
            [
                "decoder_brotli_len_le_u32",
                "decoder_brotli",
                "scales_fp16_len_le_u32",
                "scales_fp16",
                "latents_brotli_len_le_u32",
                "latents_brotli",
                "wrap_sidecar_brotli_len_le_u32",
                "wrap_sidecar_brotli",
            ],
            False,
        ),
        (
            _zip_payload(_pr107_payload(), member_name="0.bin"),
            "pr107_apogee_length_prefixed",
            [
                "meta_brotli_len_le_u32",
                "meta_brotli",
                "decoder_blob_len_le_u32",
                "decoder_blob",
                "latents_brotli_len_le_u32",
                "latents_brotli",
            ],
            False,
        ),
    ],
)
def test_detect_archive_grammar_and_parse_synthetic_boundaries(
    archive_bytes,
    expected_grammar,
    section_names,
    projection_supported,
):
    grammar, layout = emg.detect_archive_grammar_and_parse(archive_bytes)

    assert grammar == expected_grammar
    assert layout.grammar_name == expected_grammar
    assert layout.archive_bytes == len(archive_bytes)
    assert layout.gradient_byte_domain == "zip_inner_member_payload"
    assert layout.gradient_projection_supported is projection_supported
    assert [section.name for section in layout.sections] == section_names
    assert layout.sections[0].offset == 0
    assert layout.sections[-1].end_offset == layout.gradient_subject_bytes
    assert layout.as_dict()["grammar_name"] == expected_grammar


def test_detect_archive_grammar_preserves_raw_domain_for_unzipped_payload():
    grammar, layout = emg.detect_archive_grammar_and_parse(_pr101_payload())

    assert grammar == "pr101_lc_v2"
    assert layout.member_name is None
    assert layout.gradient_byte_domain == "scored_archive_bytes"
    assert layout.archive_sha256 == layout.gradient_subject_sha256


def test_detect_archive_grammar_only_cli_prints_json(tmp_path, capsys):
    archive = tmp_path / "archive.zip"
    archive.write_bytes(_zip_payload(_pr106_payload()))

    rc = emg.main(["--archive", str(archive), "--detect-grammar-only"])

    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["grammar_name"] == "pr106_format0d"
    assert parsed["gradient_projection_supported"] is False


def test_main_fail_closes_detection_only_grammar_before_codec_import(tmp_path):
    archive = tmp_path / "archive.zip"
    archive.write_bytes(_zip_payload(_pr106_payload()))
    fake_inflate = tmp_path / "inflate.py"
    fake_inflate.write_text("# no src import should happen before fail-closed grammar gate\n")
    fake_upstream = tmp_path / "upstream"
    fake_upstream.mkdir()

    with pytest.raises(SystemExit, match="xray/detection-only"):
        emg.main(
            [
                "--archive",
                str(archive),
                "--inflate-py",
                str(fake_inflate),
                "--upstream-dir",
                str(fake_upstream),
                "--axis",
                "[diagnostic-CPU]",
                "--output-npy",
                str(tmp_path / "grad.npy"),
                "--device",
                "cpu",
            ]
        )


def test_main_requires_explicit_axis_for_anchor_emitting_path(tmp_path):
    archive = tmp_path / "archive.zip"
    archive.write_bytes(_zip_payload(_pr101_payload()))
    fake_inflate = tmp_path / "inflate.py"
    fake_inflate.write_text("# noop\n")
    fake_upstream = tmp_path / "upstream"
    fake_upstream.mkdir()

    with pytest.raises(SystemExit, match="--axis is required"):
        emg.main(
            [
                "--archive",
                str(archive),
                "--inflate-py",
                str(fake_inflate),
                "--upstream-dir",
                str(fake_upstream),
                "--output-npy",
                str(tmp_path / "grad.npy"),
                "--device",
                "cpu",
            ]
        )


def test_detect_archive_grammar_real_local_fixtures_if_present():
    fixtures = [
        (
            REPO_ROOT
            / "experiments/results/a1_cuda_cpu_drift_discriminator_v_baseline_20260509T110211Z/submission_dir/archive.zip",
            "a1_finetuned",
        ),
        (
            REPO_ROOT / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip",
            "pr101_lc_v2",
        ),
        (
            REPO_ROOT / "experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/sidecar_archive.zip",
            "pr106_format0d",
        ),
        (
            REPO_ROOT / "experiments/results/public_pr_intake_full/public_pr106_intake_20260505_auto/archive.zip",
            "pr106_ff_packed_hnerv",
        ),
        (
            REPO_ROOT / "experiments/results/public_pr100_intake_20260504_codex/archive.zip",
            "hnerv_lc_v2_length_prefixed",
        ),
        (
            REPO_ROOT / "experiments/results/pr107_apogee_cpu_auth_eval_gha_20260508T124452Z/archive.zip",
            "pr107_apogee_length_prefixed",
        ),
        (
            REPO_ROOT / "experiments/results/public_pr_intake_full/public_pr107_intake_20260505_auto/archive.zip",
            "pr107_apogee_length_prefixed",
        ),
    ]
    existing = [(path, grammar) for path, grammar in fixtures if path.exists()]
    if not existing:
        pytest.skip("local production-size archive fixtures are not present")

    for path, expected_grammar in existing:
        grammar, layout = emg.detect_archive_grammar_and_parse(path.read_bytes())
        assert grammar == expected_grammar
        assert layout.gradient_subject_bytes > 100_000
        assert layout.sections[-1].end_offset == layout.gradient_subject_bytes


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


def test_maybe_extract_inner_archive_from_zip_refuses_multi_member_zip(tmp_path):
    import zipfile
    z = tmp_path / "archive.zip"
    inner = b"the canonical archive"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("0.bin", inner)
        zf.writestr("metadata.json", b"{}")
    with pytest.raises(ValueError, match="exactly one payload member"):
        emg._maybe_extract_inner_archive_from_zip(z)


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
