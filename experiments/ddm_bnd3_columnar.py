#!/usr/bin/env python3
"""Charge bnd3's joint/skip geometry using bnd2's planar varint transform.

Research-only adapter. The primary fixed-width wire defines the joint solver's
objective; this additional invertible serialization removes that wire-layout
confound from comparison with bnd2. Both real compressed forms are retained.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import brotli

from experiments import ddm_bnd3_address_term as run


def pack(frames, minimum):
    counts, starts, lengths, turns, types = (bytearray() for _ in range(5))
    for segments in frames:
        selected = sorted((s for s in segments if len(s[3]) >= minimum), key=lambda s: (s[0], s[1], s[2]))
        counts.extend(run.old.uvarint(len(selected)))
        previous = 0
        for (a, b, normal), vertex, directions, _positions, skipped in selected:
            types.append((a * 5 + b) * 3 + normal + 1)
            delta = vertex - previous
            starts.extend(run.old.uvarint(2 * delta if delta >= 0 else -2 * delta - 1))
            previous = vertex
            lengths.extend(run.old.uvarint(len(directions)))
            direction = 0
            for d, skip in zip(directions, skipped, strict=True):
                turns.append((d - direction) % 4 + 4 * int(skip))
                direction = d
    sections = [bytes(x) for x in (counts, starts, lengths, turns)]
    address = struct.pack("<4I", *(len(x) for x in sections)) + b"".join(sections)
    content = bytes(types)
    ac, cc = brotli.compress(address, quality=11, lgwin=22), brotli.compress(content, quality=11, lgwin=22)
    packet = run.HEADER.pack(b"BA3P", len(frames), run.old.H, run.old.W, len(ac), len(cc)) + ac + cc
    return {"address.raw": address, "content.raw": content, "address.br": ac, "content.br": cc, "segments": packet}


def unpack(packet):
    if len(packet) < run.HEADER.size:
        raise ValueError("short planar packet")
    magic, n, h, w, alen, clen = run.HEADER.unpack_from(packet)
    if (
        magic != b"BA3P"
        or not 1 <= n <= 600
        or (h, w) != (run.old.H, run.old.W)
        or run.HEADER.size + alen + clen != len(packet)
    ):
        raise ValueError("invalid planar header")
    raw = brotli.decompress(packet[run.HEADER.size : run.HEADER.size + alen])
    content = brotli.decompress(packet[run.HEADER.size + alen :])
    sizes = struct.unpack_from("<4I", raw)
    if 16 + sum(sizes) != len(raw):
        raise ValueError("planar section sizes differ")
    cursor, sections = 16, []
    for size in sizes:
        sections.append(raw[cursor : cursor + size])
        cursor += size
    counts, starts, lengths, turns = sections
    ci = si = li = di = ti = 0
    frames = []
    for _frame in range(n):
        count, ci = run.old.readvarint(counts, ci)
        previous, supplied = 0, {}
        for _segment in range(count):
            zz, si = run.old.readvarint(starts, si)
            vertex = previous + (zz // 2 if zz % 2 == 0 else -(zz // 2) - 1)
            previous = vertex
            length, li = run.old.readvarint(lengths, li)
            if length < 1 or di + length > len(turns) or ti >= len(content):
                raise ValueError("invalid planar chain")
            pair, normal = divmod(content[ti], 3)
            a, b = divmod(pair, 5)
            normal -= 1
            ti += 1
            if a >= 5 or a == b:
                raise ValueError("invalid planar class pair")
            direction = 0
            for value in turns[di : di + length]:
                if value > 7:
                    raise ValueError("invalid turn/skip")
                direction = (direction + value % 4) % 4
                y, x = divmod(vertex, run.old.W + 1)
                dy, dx = run.old.DIRECTIONS[direction]
                if not (0 <= y <= h and 0 <= x <= w and 0 <= y + dy <= h and 0 <= x + dx <= w):
                    raise ValueError("invalid planar crack")
                py, px = run.old.sample_edge(y, x, direction, normal)
                if not (0 <= py < h and 0 <= px < w):
                    raise ValueError("invalid planar normal")
                if value < 4:
                    pos = py * w + px
                    if pos in supplied:
                        raise ValueError("duplicate planar token")
                    supplied[pos] = b if normal == -1 else a
                vertex = run.joint.end_vertex(vertex, direction)
            di += length
        frames.append(supplied)
    if (ci, si, li, di, ti) != tuple(len(s) for s in (*sections, content)):
        raise ValueError("trailing planar data")
    return frames


BASE_BIND = run.bind


def bind(stage):
    base, library = BASE_BIND(stage)
    adapter = run.old.jg2.file_fact(Path(__file__))
    path = run.ROOT / stage / "ADAPTER.json"
    if path.exists() and json.loads(path.read_text()) != adapter:
        raise ValueError("planar adapter changed")
    run.record(path, adapter)
    return {**base, "columnar_adapter": adapter}, library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--orientation", type=int, choices=(0, 1), required=True)
    parser.add_argument("--gap", type=int, choices=(0, 1, 2, 4), required=True)
    parser.add_argument("--minimum", type=int, choices=(1, 2, 4), required=True)
    args = parser.parse_args()
    if args.resume_from.resolve() != run.ROOT.resolve():
        raise ValueError("wrong owned resume root")
    frames = []
    base, _ = BASE_BIND(f"geometry_o{args.orientation}_k{args.gap}")
    for frame in range(600):
        path = run.ROOT / f"geometry_o{args.orientation}_k{args.gap}" / f"frame_{frame:04d}.json"
        data = json.loads(path.read_text())
        if data["binding"] != base or not data["twin_identical"]:
            raise ValueError("geometry source binding failed")
        frames.append(data["segments"])
    # Explicit dependency replacement: the shared kernel prices and checks this
    # packet's actual decoder, not a cached primary-format support map.
    run.pack, run.unpack, run.bind = pack, unpack, bind
    result = run.encode(f"planar_o{args.orientation}_k{args.gap}_m{args.minimum}", frames, args.minimum, False, 600)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
