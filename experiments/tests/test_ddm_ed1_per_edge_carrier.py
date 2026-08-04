# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from experiments import ddm_ed1_per_edge_carrier as ed1


def test_bool_bitpack_padding_is_checked() -> None:
    packed = ed1.pack_bool_bits(np.array([1, 0, 1], dtype=bool))
    assert ed1.unpack_bool_bits(packed, 3).tolist() == [True, False, True]
    corrupted = bytes([packed[0] | 0b0001])
    with pytest.raises(ed1.Ed1CarrierError, match="nonzero padding"):
        ed1.unpack_bool_bits(corrupted, 3)


def test_ed1_section_reconstructs_only_band_captured_road_lane_targets() -> None:
    gt = np.full((2, 32, 40), 2, dtype=np.uint8)
    current = gt.copy()
    gt[:, 4:28, 20] = ed1.LANE_CLASS
    current[:, 4:28, 20] = ed1.LANE_CLASS

    # Two real Road/Lane disagreements inside the centerline band.
    gt[0, 10, 19] = ed1.LANE_CLASS
    current[0, 10, 19] = ed1.ROAD_CLASS
    gt[1, 18, 21] = ed1.ROAD_CLASS
    current[1, 18, 21] = ed1.LANE_CLASS

    # A Road/Lane disagreement far from the centerline is a target but not captured
    # by this radius-1 separatrix chart.
    gt[1, 2, 2] = ed1.LANE_CLASS
    current[1, 2, 2] = ed1.ROAD_CLASS

    built = ed1.build_ed1_section_from_argmax(gt, current, degree=1, band_radius=1)
    parsed = ed1.parse_ed1_section(built.section)
    maps = ed1.correction_maps_from_parsed(parsed)

    assert built.total_road_lane_targets == 3
    assert built.captured_targets == 2
    assert maps[0, 10, 19] == 2  # paint Lane
    assert maps[1, 18, 21] == 1  # paint Road
    assert maps[1, 2, 2] == 0


def test_ed1_section_refuses_corrupt_coded_payload() -> None:
    gt = np.full((1, 24, 24), 2, dtype=np.uint8)
    current = gt.copy()
    gt[0, 4:20, 12] = ed1.LANE_CLASS
    current[0, 4:20, 12] = ed1.LANE_CLASS
    gt[0, 8, 11] = ed1.LANE_CLASS
    current[0, 8, 11] = ed1.ROAD_CLASS

    built = ed1.build_ed1_section_from_argmax(gt, current, degree=1, band_radius=1)
    broken = bytearray(built.section)
    broken[-1] ^= 0x01
    with pytest.raises(Exception):
        ed1.parse_ed1_section(bytes(broken))


def test_runtime_patch_consumes_sixth_section_shape() -> None:
    base = """
def _f0pr_parse(x):
    return 4, 384, 512, {}

class Decoder:
    def __init__(self):
        self._f0_repair = None          # (coefs, atoms, seg_h, seg_w) when F0PR1 ships

    def _read_ix2(self, blob):
        if len(sections) == len(IX2_JOINT_ORDER) + 1:
            # v5 joint group: the 5th section is the F0PR1 frame_0 pose-repair stream.
            config, renderer, selector, pose_warp, f0pr = sections
            k, seg_h, seg_w, coefs = _f0pr_parse(f0pr)
            self._f0_repair = (coefs, _f0pr_dct_atoms(k, seg_h, seg_w), seg_h, seg_w)
        elif len(sections) == len(IX2_JOINT_ORDER):
            config, renderer, selector, pose_warp = sections
        else:
            raise SystemExit(
                f"ix2 container holds {len(sections)} sections, "
                f"expected {len(IX2_JOINT_ORDER)} or {len(IX2_JOINT_ORDER) + 1}")

    def f1(self, i: int) -> np.ndarray:
        return render_frame1_camera_uint8(self.packet, i)
"""
    patched = ed1._patch_inflate_runner(base)
    assert "_ed1_parse(ed1)" in patched
    assert "len(IX2_JOINT_ORDER) + 2" in patched
    assert "_ed1_apply(frame, self._ed1, i)" in patched
