from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from experiments import ddm_ec1_event_coordinate_producer as ec1
from experiments import ddm_js7_acceptance_sweep_and_compose as js7
from experiments import ddm_js7_ec1_overlay_runtime as overlay


def event(frame: int, source: int, target: int, indices: list[int]) -> bytes:
    return ec1.proposal_payload(
        frame,
        source,
        target,
        np.asarray(indices, dtype=np.int64),
        ec1.EVENT_TYPE["boundary_offset"],
    )


def test_all_real_coder_packets_decode_the_same_events() -> None:
    payloads = [event(7, 2, 0, [10]), event(53, 0, 1, [20, 21])]
    raw, candidates = overlay.build_packet_candidates(payloads)
    assert raw == b"".join(payloads)
    for coder, packet in candidates.items():
        decoded, report = overlay.decode_packet(packet)
        assert report["coder"] == coder
        assert [(row.frame, row.source_class, row.target_class, row.indices.tolist()) for row in decoded] == [
            (7, 2, 0, [10]),
            (53, 0, 1, [20, 21]),
        ]


def test_overlay_member_and_archive_split_preserve_base_prefix(tmp_path) -> None:
    _, candidates = overlay.build_packet_candidates([event(7, 2, 0, [10])])
    packet = candidates["brotli_q11"]
    member = overlay.append_overlay_member(b"base-member", packet)
    base, decoded_packet = overlay.split_overlay_member(member)
    assert base == b"base-member"
    assert decoded_packet == packet

    archive_path = tmp_path / "archive.zip"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("p", member)
    archive_path.write_bytes(stream.getvalue())
    assert overlay.read_overlay_archive(archive_path) == (b"base-member", packet)


def test_overlay_application_is_exact_and_refuses_reuse() -> None:
    payloads = [event(7, 2, 0, [10]), event(53, 0, 1, [20, 21])]
    _, candidates = overlay.build_packet_candidates(payloads)
    tokens = np.zeros((overlay.N, overlay.H, overlay.W), dtype=np.uint8)
    tokens[7].reshape(-1)[10] = 2
    report = overlay.apply_packet_inplace(tokens, candidates["raw"])
    assert report["site_count"] == 3
    assert tokens[7].reshape(-1)[10] == 0
    assert tokens[53].reshape(-1)[[20, 21]].tolist() == [1, 1]
    with pytest.raises(overlay.EC1OverlayError, match="source-class"):
        overlay.apply_packet_inplace(tokens, candidates["raw"])


def test_packet_and_footer_trailing_or_length_damage_refuses() -> None:
    _, candidates = overlay.build_packet_candidates([event(7, 2, 0, [10])])
    packet = candidates["raw"]
    damaged = packet + b"x"
    with pytest.raises(overlay.EC1OverlayError, match=r"differs|trailing"):
        overlay.decode_packet(damaged)
    member = bytearray(overlay.append_overlay_member(b"base", packet))
    member[-1] ^= 1
    with pytest.raises(overlay.EC1OverlayError, match="footer"):
        overlay.split_overlay_member(bytes(member))


def test_projected_delta_score_uses_complete_archive_bytes_and_nonlinear_pose() -> None:
    row = js7._projected_delta_score(
        robust_delta_flips=-18,
        base_pose=1.0e-5,
        pose_delta=1.0e-7,
        archive_delta_bytes=25,
    )
    expected = (
        100.0 * -18 / (600 * 384 * 512)
        + np.sqrt(10.0 * 1.01e-5)
        - np.sqrt(10.0 * 1.0e-5)
        + 25.0 * 25 / js7.RATE_DENOMINATOR
    )
    assert row["total"] == pytest.approx(expected)


def test_runtime_patches_wire_counted_overlay_into_parse_and_render() -> None:
    residual = js7._patch_runtime_source(
        (js7.CP135_RUNTIME / "runtime/residual_archive.py").read_text(), kind="residual"
    )
    f26 = js7._patch_runtime_source(
        (js7.CP135_RUNTIME / "runtime/f26_inflate.py").read_text(), kind="f26"
    )
    assert "split_overlay_member(outer)" in residual
    assert "apply_packet_inplace(tokens, overlay_packet)" in f26
    assert '"ec1_overlay": overlay_report' in f26


def test_deterministic_zip_is_repeatable_and_stored() -> None:
    first = js7._deterministic_zip(b"payload")
    second = js7._deterministic_zip(b"payload")
    assert first == second
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.namelist() == ["p"]
        assert archive.getinfo("p").compress_type == zipfile.ZIP_STORED
        assert archive.read("p") == b"payload"
