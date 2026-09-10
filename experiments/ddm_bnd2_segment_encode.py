#!/usr/bin/env python3
"""Charged unit-crack segment grammar plus unchanged causal TC1 remainder.

Research only; real n600 encode prices, no scorer claim. Local Euclidean normals
are cardinal because the cracks are unit grid edges, unlike a global row chart.
The sidecar itself tells the decoder which tokens to insert at each HPAC group.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import struct
import sys
import zlib
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

import brotli
import numpy as np

from experiments import ddm_bnd2_causal_trace as trace
from experiments import ddm_jg2_tail_reencode as jg2

ROOT = trace.ROOT / "generation2"
H, W, N = 384, 512, 600
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
RIGHT = ((1, 0), (0, -1), (-1, 0), (0, 1))
START_FROM_RIGHT_CELL = ((0, 0), (0, 1), (1, 1), (1, 0))


def uvarint(value: int) -> bytes:
    if value < 0:
        raise ValueError("unsigned integer is negative")
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def readvarint(data: bytes, offset: int) -> tuple[int, int]:
    value, start = 0, offset
    for shift in range(0, 64, 7):
        if offset >= len(data):
            raise ValueError("truncated varint")
        item = data[offset]
        offset += 1
        value |= (item & 127) << shift
        if item < 128:
            if data[start:offset] != uvarint(value):
                raise ValueError("nonminimal varint")
            return value, offset
    raise ValueError("varint overflow")


def sample_edge(y: int, x: int, direction: int, normal: int) -> tuple[int, int]:
    """Cell at signed integer offset from the directed crack's right cell."""
    sy, sx = START_FROM_RIGHT_CELL[direction]
    ny, nx = RIGHT[direction]
    return y - sy + normal * ny, x - sx + normal * nx


def frame_segments(plane: np.ndarray, best: np.ndarray, misses: np.ndarray, orientation: int) -> tuple[list, dict]:
    """Assign misses to actual class-pair cracks, then maximal directed paths.

    Orient every crack with the lower (mode 0) or higher (mode 1) class on
    its right. n=-1 selects the left cell/class; n=0 the right cell/class;
    n=+1 one more cell inward on the right. No free GT graph is used. Gaps,
    class changes, offsets and branches end a segment and incur a fresh start.
    """
    buckets = defaultdict(dict)
    eligible = []
    for pos in misses.tolist():
        y, x = divmod(pos, W)
        symbol = int(plane[y, x])
        candidates = []
        for normal in (0, -1, 1):
            for direction, (ny, nx) in enumerate(RIGHT):
                ry, rx = y - normal * ny, x - normal * nx
                ly, lx = ry - ny, rx - nx
                if not (0 <= ry < H and 0 <= rx < W and 0 <= ly < H and 0 <= lx < W):
                    continue
                a, b = int(plane[ry, rx]), int(plane[ly, lx])
                if a == b or (a < b) != (orientation == 0):
                    continue
                if (b if normal == -1 else a) != symbol:
                    continue
                sy, sx = START_FROM_RIGHT_CELL[direction]
                vertex = (ry + sy) * (W + 1) + rx + sx
                other = a if normal == -1 else b
                priority = (other != int(best[pos]), abs(normal), vertex, direction)
                candidates.append((priority, (a, b, normal), (vertex, direction)))
        if candidates:
            _, key, edge = min(candidates)
            if edge in buckets[key]:
                raise ValueError("two token locations map to one edge/offset")
            buckets[key][edge] = pos
            eligible.append(pos)
    segments = []
    for key, edges in sorted(buckets.items()):
        outgoing, incoming = defaultdict(list), defaultdict(list)
        for edge in edges:
            start, direction = edge
            dy, dx = DIRECTIONS[direction]
            end = start + dy * (W + 1) + dx
            outgoing[start].append(edge)
            incoming[end].append(edge)
        unseen = set(edges)
        starts = sorted(e for e in edges if len(incoming[e[0]]) != 1 or len(outgoing[e[0]]) != 1)
        starts.extend(sorted(edges))  # deterministic cycle entry after all open paths
        for edge in starts:
            if edge not in unseen:
                continue
            initial = edge[0]
            directions, positions = [], []
            while edge in unseen:
                unseen.remove(edge)
                vertex, direction = edge
                directions.append(direction)
                positions.append(edges[edge])
                dy, dx = DIRECTIONS[direction]
                end = vertex + dy * (W + 1) + dx
                if len(outgoing[end]) != 1 or len(incoming[end]) != 1:
                    break
                edge = outgoing[end][0]
            segments.append((key, initial, directions, positions))
    if sorted(p for s in segments for p in s[3]) != sorted(eligible):
        raise ValueError("segment partition loses or duplicates eligible tokens")
    return segments, {"misses": len(misses), "eligible": len(eligible), "segments": len(segments)}


def pack_segments(frames: list[list], minimum: int) -> tuple[bytes, bytes]:
    """Real context transform: constant pair/offset per chain and previous-direction turns.

    All five byte sections share one Brotli context/dictionary stream. Counts,
    starts, class pairs, normals, lengths and turns are all charged.
    """
    counts, types, starts, lengths, turns = (bytearray() for _ in range(5))
    for segments in frames:
        selected = [s for s in segments if len(s[2]) >= minimum]
        counts.extend(uvarint(len(selected)))
        previous_start = 0
        for (a, b, normal), start, directions, _positions in selected:
            types.append((a * 5 + b) * 3 + normal + 1)
            delta = start - previous_start
            starts.extend(uvarint(2 * delta if delta >= 0 else -2 * delta - 1))
            previous_start = start
            lengths.extend(uvarint(len(directions)))
            previous_direction = 0
            for direction in directions:
                turns.append((direction - previous_direction) % 4)
                previous_direction = direction
    sections = [bytes(x) for x in (counts, types, starts, lengths, turns)]
    raw = b"BSG2" + struct.pack("<HHH5I", len(frames), H, W, *(len(x) for x in sections)) + b"".join(sections)
    compressed = brotli.compress(raw, quality=11, lgwin=22)
    packet = b"BSC2" + struct.pack("<II", len(raw), zlib.crc32(raw)) + compressed
    return raw, packet


def unpack_segments(packet: bytes, *, with_edges: bool = False):
    """Parse only the charged packet; no source field, trace or GT is an input."""
    if packet[:4] != b"BSC2" or len(packet) < 12:
        raise ValueError("invalid segment packet")
    size, crc = struct.unpack_from("<II", packet, 4)
    if size > 64 * 1024**2:
        raise ValueError("oversized segment packet")
    raw = brotli.decompress(packet[12:])
    if len(raw) != size or zlib.crc32(raw) != crc or raw[:4] != b"BSG2":
        raise ValueError("segment size, checksum or magic mismatch")
    n, h, w, *sizes = struct.unpack_from("<HHH5I", raw, 4)
    offset = struct.calcsize("<HHH5I") + 4
    if not 1 <= n <= N or (h, w) != (H, W) or sum(sizes) + offset != len(raw):
        raise ValueError("segment framing or dimensions mismatch")
    sections = []
    for length in sizes:
        sections.append(raw[offset : offset + length])
        offset += length
    counts, types, starts, lengths, turns = sections
    ci = ti = si = li = di = 0
    frames, frame_edges = [], []
    for _frame in range(n):
        count, ci = readvarint(counts, ci)
        previous_start = 0
        supplied = {}
        decoded_edges = []
        edges_seen = set()
        for _segment in range(count):
            if ti >= len(types):
                raise ValueError("missing segment type")
            pair, normal_code = divmod(types[ti], 3)
            a, b = divmod(pair, 5)
            normal = normal_code - 1
            ti += 1
            if not (0 <= a < 5 and 0 <= b < 5 and a != b):
                raise ValueError("invalid class pair")
            zz, si = readvarint(starts, si)
            start = previous_start + (zz // 2 if zz % 2 == 0 else -(zz // 2) - 1)
            previous_start = start
            length, li = readvarint(lengths, li)
            if length < 1 or di + length > len(turns):
                raise ValueError("invalid chain length")
            y, x = divmod(start, W + 1)
            direction = 0
            for turn in turns[di : di + length]:
                if turn > 3:
                    raise ValueError("invalid turn code")
                direction = (direction + turn) % 4
                dy, dx = DIRECTIONS[direction]
                ey, ex = y + dy, x + dx
                if not (0 <= y <= H and 0 <= x <= W and 0 <= ey <= H and 0 <= ex <= W):
                    raise ValueError("crack vertex outside image")
                edge = (a, b, normal, y, x, direction)
                if edge in edges_seen:
                    raise ValueError("duplicate crack at the same offset")
                edges_seen.add(edge)
                decoded_edges.append((y, x, direction, a, b))
                py, px = sample_edge(y, x, direction, normal)
                if not (0 <= py < H and 0 <= px < W):
                    raise ValueError("normal raster outside image")
                pos = py * W + px
                if pos in supplied:
                    raise ValueError("duplicate supplied cell")
                supplied[pos] = b if normal == -1 else a
                y, x = ey, ex
            di += length
        frames.append(supplied)
        frame_edges.append(decoded_edges)
    if (ci, ti, si, li, di) != tuple(len(s) for s in sections):
        raise ValueError("trailing segment symbols")
    return (frames, frame_edges) if with_edges else frames


def verify_crack_sides(plane: np.ndarray, edges: list) -> None:
    """Verify every claimed interface on the independently reconstructed field."""
    for y, x, direction, a, b in edges:
        ry, rx = sample_edge(y, x, direction, 0)
        ly, lx = sample_edge(y, x, direction, -1)
        if not (0 <= ry < H and 0 <= rx < W and 0 <= ly < H and 0 <= lx < W):
            raise ValueError("claimed interface lacks two in-image sides")
        if int(plane[ry, rx]) != a or int(plane[ly, lx]) != b:
            raise ValueError("decoded crack classes differ from reconstructed sides")


def composite(segment: bytes, residual: bytes) -> bytes:
    if residual[:4] != b"R6D1" or residual[-1:] != b"\0":
        raise ValueError("unexpected RC64 envelope")
    return b"BND2" + struct.pack("<I", len(segment)) + segment + residual[4:]


def split_composite(payload: bytes) -> tuple[list[dict[int, int]], bytes]:
    if payload[:4] != b"BND2" or len(payload) < 9:
        raise ValueError("invalid replacement envelope")
    length = struct.unpack_from("<I", payload, 4)[0]
    if length + 9 > len(payload):
        raise ValueError("truncated replacement envelope")
    return unpack_segments(payload[8 : 8 + length]), b"R6D1" + payload[8 + length :]


def complete_trace() -> dict:
    result = json.loads((ROOT / "trace/RESULT.json").read_text())
    if result["frames"] != N or not result["full_control_byte_identical"] or result["field_sha256"] != trace.FIELD_SHA:
        raise ValueError("pricing requires a complete identity-verified shipped trace")
    for fact in result["payloads"]:
        if jg2.file_fact(Path(fact["path"])) != fact:
            raise ValueError("source trace envelope custody changed")
    return result


def load_frame(frame: int) -> dict:
    path = ROOT / "trace/frames" / f"frame_{frame:04d}.npz"
    receipt = json.loads(path.with_suffix(".json").read_text())
    if receipt["source_field_sha256"] != trace.FIELD_SHA or jg2.file_fact(path) != receipt["payload"]:
        raise ValueError("trace frame source/hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def price(orientation: int, minimum: int) -> dict:
    """Full-n600 twin arithmetic encodes, then actual full-field parseback."""
    source = complete_trace()
    work = ROOT / f"price_o{orientation}_min{minimum}"
    trace.storage(work / "build")
    route = jg2.load_route_b()
    # Reuse the trace's pinned native binary; no new compiler state at resume.
    library = Path(source["build"]["library"]["path"])
    if jg2.file_fact(library) != source["build"]["library"]:
        raise ValueError("native library custody changed")
    binding = {
        "source": jg2.sha256_file(ROOT / "trace/RESULT.json"),
        "producer": jg2.sha256_file(Path(__file__)),
        "trace_helper": jg2.sha256_file(Path(trace.__file__)),
        "route": jg2.sha256_file(Path(route.__file__)),
        "orientation": orientation,
        "minimum": minimum,
    }
    grammar_path = work / "GRAMMAR.json"
    if grammar_path.exists():
        grammar = json.loads(grammar_path.read_text())
        if grammar["binding"] != binding:
            raise ValueError("grammar resume binding differs")
        for fact in grammar["payloads"]:
            if jg2.file_fact(Path(fact["path"])) != fact:
                raise ValueError("retained grammar changed")
        segment = (work / "primary.segments").read_bytes()
    else:
        frames = []
        run_lengths = []
        census = []
        for frame in range(N):
            geometry_path = ROOT / f"geometry_o{orientation}" / f"frame_{frame:04d}.json"
            geometry_binding = {
                "source": binding["source"],
                "producer": binding["producer"],
                "orientation": orientation,
            }
            if geometry_path.exists():
                geometry = json.loads(geometry_path.read_text())
                if geometry["binding"] != geometry_binding:
                    raise ValueError("geometry checkpoint binding changed")
                segments, counts = geometry["segments"], geometry["counts"]
            else:
                data = load_frame(frame)
                segments, counts = frame_segments(data["tokens"], data["best"], data["miss_pos"], orientation)
                trace.record(geometry_path, {"binding": geometry_binding, "segments": segments, "counts": counts})
            frames.append(segments)
            run_lengths.extend(len(s[2]) for s in segments)
            census.append({"frame": frame, **counts})
        raw0, segment = pack_segments(frames, minimum)
        # Independent serialization rebuild, not copying the first payload.
        raw1, repeat = pack_segments(frames, minimum)
        facts = [
            trace.blob(work / name, value)
            for name, value in (
                ("primary.raw", raw0),
                ("repeat.raw", raw1),
                ("primary.segments", segment),
                ("repeat.segments", repeat),
            )
        ]
        if (raw0, segment) != (raw1, repeat):
            raise ValueError("segment twin encoding differs")
        supplied = unpack_segments(segment)
        for frame, segments in enumerate(frames):
            expected = {
                p: (s[0][1] if s[0][2] == -1 else s[0][0]) for s in segments if len(s[2]) >= minimum for p in s[3]
            }
            if supplied[frame] != expected:
                raise ValueError("charged grammar parseback lost selected tokens")
        grammar = trace.record(
            grammar_path,
            {
                "binding": binding,
                "payloads": facts,
                "census": census,
                "run_lengths": run_lengths,
                "supplied_tokens": sum(len(f) for f in supplied),
                "segment_twin_identical": True,
                "segment_parseback_exact": True,
            },
        )
    supplied = unpack_segments(segment)
    if len(supplied) != N:
        raise ValueError("pricing requires all 600 segment planes")
    order = np.frombuffer((ROOT / "trace/group_order.i32").read_bytes(), dtype="<i4")
    latest = work / "LATEST.json"
    start, state = 0, None
    if latest.exists():
        receipt = json.loads(latest.read_text())
        if receipt["binding"] != binding or jg2.file_fact(Path(receipt["payload"]["path"])) != receipt["payload"]:
            raise ValueError("residual checkpoint changed")
        with np.load(receipt["payload"]["path"], allow_pickle=False) as data:
            state = {k: data[k] for k in data.files}
        start = int(state["frame"][0])
    encoders = [
        route.NativeRc64Encoder(library, None if state is None else state[k].tobytes())
        for k in ("encoder0", "encoder1")
    ]
    for frame in range(start, N):
        data = load_frame(frame)
        truth = data["tokens"].reshape(-1)
        mask = np.ones(H * W, dtype=bool)
        misses = set(data["miss_pos"].tolist())
        for pos, symbol in supplied[frame].items():
            if truth[pos] != symbol or pos not in misses:
                raise ValueError("segment supplies a non-miss or wrong shipped symbol")
            mask[pos] = False
        positions = order[mask[order]]
        for encoder in encoders:
            encoder.encode(truth[positions].astype(np.int32), data["rows"][positions])
        if (frame + 1) % 20 == 0 or frame + 1 == N:
            values = {
                "frame": np.array([frame + 1]),
                "encoder0": np.frombuffer(encoders[0].snapshot(), dtype=np.uint8),
                "encoder1": np.frombuffer(encoders[1].snapshot(), dtype=np.uint8),
            }
            fact = trace.arrays(work / "checkpoints" / f"encode_{frame + 1:04d}.npz", values)
            trace.record(latest, {"binding": binding, "payload": fact})
            print(json.dumps({"stage": work.name, "frame": frame + 1}), flush=True)
    residuals = []
    for tag, encoder in zip(("primary", "repeat"), encoders, strict=False):
        native_envelope = encoder.finish()
        trace.blob(work / f"{tag}.native.envelope", native_envelope)
        size = int(encoder.library.rc64_encoder_size(encoder.context))
        raw = ctypes.string_at(encoder.library.rc64_encoder_data(encoder.context), size)
        trace.blob(work / f"{tag}.residual.raw", raw)
        # The experiment envelope uses exactly one zero byte, like the pinned
        # baseline, rather than route_b's variable 0..3 byte u32 alignment pad.
        residuals.append(b"R6D1" + raw + b"\0")
    for name, value in zip(("primary", "repeat"), residuals, strict=False):
        trace.blob(work / f"{name}.residual.envelope", value)
        trace.blob(work / f"{name}.replacement.envelope", composite(segment, value))
    if residuals[0] != residuals[1]:
        raise ValueError("residual twins differ")
    envelope = (work / "primary.replacement.envelope").read_bytes()
    decoded_support, residual = split_composite(envelope)
    decode_latest = work / "DECODE_LATEST.json"
    decode_start = 0
    if decode_latest.exists():
        receipt = json.loads(decode_latest.read_text())
        if receipt["binding"] != binding or jg2.file_fact(Path(receipt["payload"]["path"])) != receipt["payload"]:
            raise ValueError("decode checkpoint changed")
        decoder = route.NativeRc64Decoder(library, Path(receipt["payload"]["path"]).read_bytes())
        decode_start = receipt["frame"]
    else:
        decoder = route.NativeRc64Decoder(library, residual)
    for frame in range(decode_start, N):
        data = load_frame(frame)
        restored = np.zeros(H * W, dtype=np.uint8)
        mask = np.ones(H * W, dtype=bool)
        for pos, symbol in decoded_support[frame].items():
            mask[pos] = False
            restored[pos] = symbol
        positions = order[mask[order]]
        restored[positions] = decoder.decode(None, data["rows"][positions]).astype(np.uint8)
        trace.blob(work / "decoded" / f"frame_{frame:04d}.u8", restored.tobytes())
        if not np.array_equal(restored, data["tokens"].reshape(-1)):
            raise ValueError("segment plus residual parseback failed")
        if (frame + 1) % 20 == 0 or frame + 1 == N:
            checkpoint = decoder.get_compressed().tobytes()
            fact = trace.blob(work / "checkpoints" / f"decode_{frame + 1:04d}.bin", checkpoint)
            trace.record(decode_latest, {"binding": binding, "payload": fact, "frame": frame + 1})
    digest = hashlib.sha256()
    for frame in range(N):
        decoded_plane = (work / "decoded" / f"frame_{frame:04d}.u8").read_bytes()
        if len(decoded_plane) != H * W:
            raise ValueError("retained decoded plane length changed")
        digest.update(decoded_plane)
    if digest.hexdigest() != trace.FIELD_SHA:
        raise ValueError("full reconstructed field hash differs")
    old_member = jg2.read_archive_member(ROOT / "runtime/archive.zip")
    old_raw = (trace.ROOT / "retained/shipped_token_stream.rc64").read_bytes()
    weights = (trace.ROOT / "retained/tc1_weights.bin").read_bytes()
    old_rider = b"TC1M\x01" + weights + old_raw
    if not old_member.endswith(old_rider):
        raise ValueError("cmp2 container tail is not the expected unchanged TC1 rider")
    new_member = old_member[: -len(old_rider)] + b"BND2\x01" + weights + envelope[4:-1]
    archive = work / "archive.zip"
    trace.storage(archive, len(new_member))
    if not archive.exists():
        jg2.pack_archive(new_member, archive)
    if jg2.read_archive_member(archive) != new_member:
        raise ValueError("ZIP member parseback mismatch")
    repeat_envelope = (work / "repeat.replacement.envelope").read_bytes()
    repeat_member = old_member[: -len(old_rider)] + b"BND2\x01" + weights + repeat_envelope[4:-1]
    repeat_archive = work / "archive.repeat.zip"
    if not repeat_archive.exists():
        jg2.pack_archive(repeat_member, repeat_archive)
    if jg2.read_archive_member(repeat_archive) != repeat_member or repeat_archive.read_bytes() != archive.read_bytes():
        raise ValueError("full candidate archive twins differ")
    fact = jg2.file_fact(archive)
    if fact["bytes"] - 180388 != len(envelope) - 119784:
        raise ValueError("full container byte accounting differs")
    return trace.record(
        work / "RESULT.json",
        {
            "schema": "boundary_segment_recode_price_v1",
            "axis": trace.AXIS,
            "score_claim": False,
            "binding": binding,
            "frames": N,
            "symbols": N * H * W,
            "segment_bytes_including_length": 4 + len(segment),
            "residual_envelope_bytes": len(residual),
            "total_envelope_bytes": len(envelope),
            "baseline_envelope_bytes": 119784,
            "delta_bytes": len(envelope) - 119784,
            "archive": fact,
            "repeat_archive": jg2.file_fact(repeat_archive),
            "replacement": jg2.file_fact(work / "primary.replacement.envelope"),
            "twin_byte_identical": True,
            "cached_rows_parseback_exact": True,
            "decoded_field_sha256": digest.hexdigest(),
            "source_field_sha256": trace.FIELD_SHA,
            "independent_causal_decoder": "not yet run",
            "supplied_tokens": grammar["supplied_tokens"],
            "residual_tokens": N * H * W - grammar["supplied_tokens"],
            "draw_gate_bytes_pass": len(envelope) <= 114784,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--orientation", type=int, choices=(0, 1), default=0)
    parser.add_argument("--minimum-run", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    if args.resume_from.resolve() != trace.ROOT.resolve():
        raise ValueError("resume root outside owned store")
    print(json.dumps(price(args.orientation, args.minimum_run), sort_keys=True))


if __name__ == "__main__":
    main()
