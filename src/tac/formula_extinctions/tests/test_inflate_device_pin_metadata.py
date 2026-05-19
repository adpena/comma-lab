# SPDX-License-Identifier: MIT
"""Tests for Row #4 — Inflate device pin metadata."""
from __future__ import annotations

import pytest

from tac.formula_extinctions.inflate_device_pin_metadata import (
    InflateDevicePinInput,
    canonical_inflate_device_pin_metadata,
)


def test_linux_cpu_to_contest_cpu_tag():
    """Linux x86_64 CPU -> [contest-CPU] axis tag."""
    r = canonical_inflate_device_pin_metadata(InflateDevicePinInput(
        device="cpu", score_axis="contest_cpu", linux_x86_64_compliant=True,
    ))
    assert r.solved_value["score_axis_canonical_tag"] == "[contest-CPU]"
    assert r.intermediate_values["is_authoritative_axis"] is True


def test_macos_cpu_to_advisory_tag():
    """macOS CPU is NEVER [contest-CPU]; tag becomes [macOS-CPU advisory]."""
    r = canonical_inflate_device_pin_metadata(InflateDevicePinInput(
        device="cpu", score_axis="contest_cpu", linux_x86_64_compliant=False,
    ))
    assert r.solved_value["score_axis_canonical_tag"] == "[macOS-CPU advisory]"
    assert r.intermediate_values["is_authoritative_axis"] is False


def test_cuda_to_contest_cuda_tag():
    """CUDA on linux_x86_64-compliant hardware -> [contest-CUDA]."""
    r = canonical_inflate_device_pin_metadata(InflateDevicePinInput(
        device="cuda", score_axis="contest_cuda", linux_x86_64_compliant=True,
    ))
    assert r.solved_value["score_axis_canonical_tag"] == "[contest-CUDA]"


def test_diagnostic_axes_not_authoritative():
    """diagnostic axes are non-authoritative even on compliant hardware."""
    r = canonical_inflate_device_pin_metadata(InflateDevicePinInput(
        device="cpu", score_axis="diagnostic_cpu", linux_x86_64_compliant=True,
    ))
    assert r.intermediate_values["is_authoritative_axis"] is False
    assert r.solved_value["score_axis_canonical_tag"] == "[diagnostic-CPU]"


def test_invalid_inputs_raise():
    """Invalid device or axis raises."""
    with pytest.raises(ValueError, match="device"):
        InflateDevicePinInput(device="mps", score_axis="contest_cpu")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="score_axis"):
        InflateDevicePinInput(device="cpu", score_axis="bogus_axis")


def test_metadata_carries_archive_sha():
    """archive_sha256 is preserved in solved_value metadata dict."""
    r = canonical_inflate_device_pin_metadata(InflateDevicePinInput(
        device="cpu",
        score_axis="contest_cpu",
        linux_x86_64_compliant=True,
        archive_sha256="87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5",
    ))
    assert r.solved_value["archive_sha256"].startswith("87ec7ca5")
