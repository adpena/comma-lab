"""Guards on ddm_ren1's load-bearing pure math.

Everything tested here is a piece whose silent failure would make the arm's n600
rows mean something other than what the memo says they mean:

* the two perturbation fields must be RMS-MATCHED, or the smooth/noise ratio
  measures magnitude instead of spectrum and the whole comparison is void;
* the perturbation must be RE-QUANTISED into uint8 the way the receiver's own
  ``clamp(0,255).round()`` does, or the arm is scoring a float fiction no decode
  could emit;
* ``changed_fraction`` is the no-op detector: a half-LSB field that rounded back
  to the same bytes everywhere would give a null Pose reading that is a fact
  about rounding, not about the spectrum;
* the frame-1 shim's index arithmetic must map the solver's ``2*pair+1`` request
  to the right pair, and must REFUSE anything else rather than silently serve a
  neighbour's frame.

No torch model, no archive and no scorer is loaded: these are the arithmetic
pieces, and they are the ones that can be wrong without anything crashing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

probe = pytest.importorskip("ddm_ren1_step0_probe")
absorption = pytest.importorskip("ddm_ren1_resolve_absorption")
torch = pytest.importorskip("torch")


def test_perturbation_fields_are_rms_matched_to_the_declared_amplitude():
    smooth, noise = probe.perturbation_fields(20260911)
    for field in (smooth, noise):
        rms = float(field.square().mean().sqrt())
        assert rms == pytest.approx(probe.PERTURB_RMS_LSB, rel=1e-5)
        assert float(field.mean()) == pytest.approx(0.0, abs=1e-5)
    # Matched magnitude is the whole point; the ratio the memo quotes is only a
    # spectrum measurement if these two agree.
    smooth_rms = float(smooth.square().mean().sqrt())
    noise_rms = float(noise.square().mean().sqrt())
    assert smooth_rms == pytest.approx(noise_rms, rel=1e-5)


def test_perturbation_fields_are_camera_shaped_and_differ_in_spectrum():
    smooth, noise = probe.perturbation_fields(20260911)
    assert tuple(smooth.shape) == (probe.jg1.CAMERA_H, probe.jg1.CAMERA_W, 3)
    assert tuple(noise.shape) == (probe.jg1.CAMERA_H, probe.jg1.CAMERA_W, 3)
    # A bicubic upsample of a 24x32 draw is far smoother than an iid draw: the
    # mean absolute horizontal difference separates them by orders of magnitude.
    def roughness(field):
        return float((field[:, 1:, :] - field[:, :-1, :]).abs().mean())

    assert roughness(smooth) < roughness(noise) / 10.0


def test_perturbation_fields_are_deterministic_in_the_seed():
    first = probe.perturbation_fields(7)
    second = probe.perturbation_fields(7)
    other = probe.perturbation_fields(8)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert not torch.equal(first[0], other[0])


def test_apply_perturbation_returns_uint8_and_respects_the_receiver_clamp():
    frames = np.full((2, 4, 5, 3), 128, dtype=np.uint8)
    frames[0, 0, 0, :] = 0
    frames[0, 0, 1, :] = 255
    field = torch.full((4, 5, 3), 40.0)
    moved = probe.apply_perturbation(frames, field)
    assert moved.dtype == np.uint8
    assert moved.shape == frames.shape
    assert moved[0, 0, 0, 0] == 40  # 0 + 40
    assert moved[0, 0, 1, 0] == 255  # 255 + 40 clamps, never wraps
    assert moved[1, 3, 4, 0] == 168  # 128 + 40


def test_apply_perturbation_clamps_the_low_end_rather_than_wrapping():
    frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    moved = probe.apply_perturbation(frames, torch.full((2, 2, 3), -7.0))
    assert int(moved.min()) == 0 and int(moved.max()) == 0


def test_apply_perturbation_rounds_half_to_even_like_torch():
    frames = np.zeros((1, 1, 2, 3), dtype=np.uint8)
    frames[...] = 4
    moved = probe.apply_perturbation(frames, torch.full((1, 2, 3), 0.5))
    # torch.round is round-half-to-even, so 4.5 -> 4.  The test pins the actual
    # behaviour rather than an assumed one, because the memo's changed-fraction
    # is read off it.
    assert int(moved[0, 0, 0, 0]) == 4


def test_changed_fraction_is_the_no_op_detector():
    before = np.zeros((2, 3, 4, 3), dtype=np.uint8)
    after = before.copy()
    assert probe.changed_fraction(before, after) == 0.0
    after[0, 0, 0, 0] = 1
    assert probe.changed_fraction(before, after) == pytest.approx(1.0 / before.size)
    assert probe.changed_fraction(before, before + 1) == 1.0


def test_half_lsb_fields_actually_move_bytes():
    """A half-LSB perturbation must not round away to a no-op."""
    smooth, noise = probe.perturbation_fields(20260911)
    frames = np.full((1, probe.jg1.CAMERA_H, probe.jg1.CAMERA_W, 3), 120, dtype=np.uint8)
    for field in (smooth, noise):
        moved = probe.apply_perturbation(frames, field)
        assert probe.changed_fraction(frames, moved) > 0.2


def test_treatments_include_control_and_the_two_matched_spectra():
    assert probe.TREATMENTS[0] == "control"
    for name in ("gt_partition", "smooth_p05", "noise_p05"):
        assert name in probe.TREATMENTS


def test_retain_refuses_a_write_outside_the_probe_store(tmp_path):
    with pytest.raises(probe.Ren1ProbeError):
        probe.retain(tmp_path / "escape.json", b"{}")


class _StubRenderer:
    """Records the pair it was asked for and returns a recognisable frame."""

    def __init__(self):
        self.calls: list[int] = []


def _stub_frames(monkeypatch, treatment, field):
    renderer = _StubRenderer()

    def fake_render(_semantic, tokens_chunk, pair_indices):
        renderer.calls.append(int(pair_indices[0]))
        return np.full((1, 2, 2, 3), int(pair_indices[0]) % 251, dtype=np.uint8)

    monkeypatch.setattr(absorption.jg1, "render_frame1", fake_render)
    tokens = np.zeros((600, 1, 1), dtype=np.uint8)
    shim = absorption.TreatmentFrames(None, tokens, field, treatment)
    return shim, renderer


def test_frame_shim_maps_the_solvers_odd_index_to_the_right_pair(monkeypatch):
    shim, renderer = _stub_frames(monkeypatch, "control", None)
    frames = shim[np.array([2 * 17 + 1])]
    assert renderer.calls == [17]
    assert int(frames[0, 0, 0, 0]) == 17


def test_frame_shim_refuses_an_even_index(monkeypatch):
    shim, _ = _stub_frames(monkeypatch, "control", None)
    with pytest.raises(absorption.Ren1ResolveError):
        shim[np.array([34])]


def test_frame_shim_refuses_a_multi_pair_request(monkeypatch):
    shim, _ = _stub_frames(monkeypatch, "control", None)
    with pytest.raises(absorption.Ren1ResolveError):
        shim[np.array([1, 3])]


def test_frame_shim_applies_the_treatment_field(monkeypatch):
    field = torch.full((2, 2, 3), 10.0)
    shim, _ = _stub_frames(monkeypatch, "noise_p05", field)
    frames = shim[np.array([2 * 5 + 1])]
    assert int(frames[0, 0, 0, 0]) == 15  # 5 from the stub render, +10 from the field


def test_frame_shim_control_arm_leaves_the_render_untouched(monkeypatch):
    shim, _ = _stub_frames(monkeypatch, "control", torch.full((2, 2, 3), 10.0))
    frames = shim[np.array([2 * 5 + 1])]
    assert int(frames[0, 0, 0, 0]) == 5


def test_frame_shim_caches_one_pair_and_re_renders_on_a_new_one(monkeypatch):
    shim, renderer = _stub_frames(monkeypatch, "control", None)
    shim[np.array([3])]
    shim[np.array([3])]
    assert renderer.calls == [1]
    shim[np.array([5])]
    assert renderer.calls == [1, 2]


def test_absorption_retain_refuses_a_write_outside_its_store(tmp_path):
    with pytest.raises(absorption.Ren1ResolveError):
        absorption.retain(tmp_path / "escape.json", b"{}")


def test_the_probe_pins_the_pointer_archive_and_field_by_sha():
    assert probe.POINTER_ARCHIVE_SHA256 == (
        "d830edd371641e1968765ae6be27120a6c55a9b1ca4b3747158e603a43ef149c"
    )
    assert probe.POINTER_ARCHIVE_BYTES == 179_111
    assert probe.TOKEN_FIELD_SHA256 == (
        "a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8"
    )
