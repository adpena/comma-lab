"""Real-input implementation checks; no subset price or representation verdict."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_bnd2_causal_trace as trace
from experiments import ddm_bnd2_segment_encode as segment
from experiments import ddm_jg2_tail_reencode as jg2


def test_real_cracks_twins_mutations_and_arithmetic_resume():
    selected = np.sort(np.random.default_rng(20260910).choice(128, 32, replace=False)).tolist()
    data = [segment.load_frame(frame) for frame in selected]
    out = trace.ROOT / "generation2/validation" / hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    geometry = {}
    for orientation in (0, 1):
        frames = [segment.frame_segments(d["tokens"], d["best"], d["miss_pos"], orientation)[0] for d in data]
        geometry[orientation] = frames
        for minimum in (1, 2, 4):
            raw, packet = segment.pack_segments(frames, minimum)
            raw_repeat, repeat = segment.pack_segments(frames, minimum)
            prefix = out / f"o{orientation}_min{minimum}"
            for suffix, value in (
                ("raw", raw),
                ("repeat.raw", raw_repeat),
                ("segments", packet),
                ("repeat.segments", repeat),
            ):
                trace.blob(prefix.with_suffix("." + suffix), value)
            assert (raw, packet) == (raw_repeat, repeat)
            supplied, edges = segment.unpack_segments(packet, with_edges=True)
            for frame, current in enumerate(supplied):
                expected = {
                    p: int(data[frame]["tokens"].reshape(-1)[p])
                    for s in frames[frame]
                    if len(s[2]) >= minimum
                    for p in s[3]
                }
                assert current == expected
                segment.verify_crack_sides(data[frame]["tokens"], edges[frame])
            broken = bytearray(packet)
            broken[8] ^= 1
            trace.blob(prefix.with_suffix(".bad_crc.segments"), bytes(broken))
            with pytest.raises(ValueError):
                segment.unpack_segments(bytes(broken))
    duplicate = [list(x) for x in geometry[0]]
    first = next(i for i, f in enumerate(duplicate) if f)
    duplicate[first].append(duplicate[first][0])
    raw, bad = segment.pack_segments(duplicate, 1)
    trace.blob(out / "duplicate.raw", raw)
    trace.blob(out / "duplicate.segments", bad)
    with pytest.raises(ValueError, match="duplicate"):
        segment.unpack_segments(bad)
    _raw, packet = segment.pack_segments(geometry[0], 1)
    supplied = segment.unpack_segments(packet)
    build = __import__("json").loads((trace.ROOT / "generation2/trace/BUILD.json").read_text())
    library = Path(build["library"]["path"])
    assert jg2.file_fact(library) == build["library"]
    route = jg2.load_route_b()
    order = np.frombuffer((trace.ROOT / "generation2/trace/group_order.i32").read_bytes(), dtype="<i4")
    encoders = [route.NativeRc64Encoder(library), route.NativeRc64Encoder(library)]
    positions_list = []
    for index, d in enumerate(data):
        mask = np.ones(segment.H * segment.W, dtype=bool)
        mask[list(supplied[index])] = False
        positions = order[mask[order]]
        positions_list.append(positions)
        for encoder in encoders:
            encoder.encode(d["tokens"].reshape(-1)[positions].astype(np.int32), d["rows"][positions])
        if index == 15:
            checkpoint = encoders[1].snapshot()
            trace.blob(out / "encoder_0016.checkpoint", checkpoint)
            encoders[1].close()
            encoders[1] = route.NativeRc64Encoder(library, (out / "encoder_0016.checkpoint").read_bytes())
    payloads = [e.finish() for e in encoders]
    for tag, value in zip(("uninterrupted", "resumed"), payloads, strict=True):
        trace.blob(out / f"{tag}.envelope", value)
    assert payloads[0] == payloads[1]
    decoder = route.NativeRc64Decoder(library, payloads[0])
    for index, (d, positions) in enumerate(zip(data, positions_list, strict=True)):
        restored = np.empty(segment.H * segment.W, dtype=np.uint8)
        restored[positions] = decoder.decode(None, d["rows"][positions])
        for pos, symbol in supplied[index].items():
            restored[pos] = symbol
        trace.blob(out / f"restored_{index:04d}.u8", restored.tobytes())
        np.testing.assert_array_equal(restored, d["tokens"].reshape(-1))
        if index == 15:
            checkpoint = decoder.get_compressed().tobytes()
            trace.blob(out / "decoder_0016.checkpoint", checkpoint)
            decoder.close()
            decoder = route.NativeRc64Decoder(library, (out / "decoder_0016.checkpoint").read_bytes())
    trace.record(
        out / "RESULT.json",
        {
            "axis": "real-input implementation validation; no subset pricing verdict",
            "score_claim": False,
            "selected_frames": selected,
            "selection_scope": "first 128 available source frames",
            "seed": 20260910,
            "grammar_variants": 6,
            "grammar_twins_and_parseback": True,
            "checksum_and_duplicate_negatives_refused": True,
            "native_encoder_resume_exact": True,
            "native_decoder_resume_exact": True,
            "trace_source_field_sha256": trace.FIELD_SHA,
            "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "segment_code_sha256": hashlib.sha256(Path(segment.__file__).read_bytes()).hexdigest(),
        },
    )
