#!/usr/bin/env python3
"""Retained n600 split-address, joint-assignment and signalled-oracle encodes.

Research only, exact serialized bytes, no scorer. All bytes, including losing
payloads, remain on the owned SSD. Frame and stage states support disk resume.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import brotli
import numpy as np

from experiments import ddm_bnd2_segment_encode as old
from experiments import ddm_bnd3_joint_address as joint

ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_bnd3_address_term")
AXIS = "[exact serialized bytes; scorer-free macOS-CPU n600]"
SEED = 20260910
HEADER = struct.Struct("<4sHHHII")


def storage(path, need=16 * 1024**2):
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("write outside owned bnd3 store")
    if shutil.disk_usage(ROOT.parent).free < 40 * 1024**3 + need:
        raise RuntimeError("STORAGE_RESERVE: keep bytes, block run")
    path.parent.mkdir(parents=True, exist_ok=True)


def blob(path, payload):
    storage(path, len(payload))
    old.jg2.persist_immutable_bytes(path, payload, label="bnd3 retained payload")
    return old.jg2.file_fact(path)


def record(path, value):
    storage(path)
    old.jg2.atomic_json(path, value)
    return value


def checkpoint(path, arrays):
    storage(path, sum(v.nbytes for v in arrays.values()))
    if not path.exists():
        temporary = path.with_suffix(".new")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    else:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != set(arrays) or any(not np.array_equal(data[k], v) for k, v in arrays.items()):
                raise ValueError("checkpoint collision")
    return old.jg2.file_fact(path)


def bind(stage):
    """Read the LIVE pointer before each stage, pin move37, fail on promotion."""
    pointer = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    if pointer["our_local_frontier_contest_cuda"]["archive_sha256"] != old.trace.ARCHIVE_SHA:
        raise RuntimeError("POINTER_MOVED: current run is move37; rebind explicitly before another stage")
    source = old.complete_trace()
    library = Path(source["build"]["library"]["path"])
    if old.jg2.file_fact(library) != source["build"]["library"]:
        raise ValueError("native library changed")
    if old.jg2.sha256_file(old.ROOT / "field.u8") != old.trace.FIELD_SHA:
        raise ValueError("source field changed")
    binding = {
        "producer": old.jg2.file_fact(Path(__file__)),
        "joint": old.jg2.file_fact(Path(joint.__file__)),
        "old_codec": old.jg2.file_fact(Path(old.__file__)),
        "source": old.jg2.file_fact(old.ROOT / "trace/RESULT.json"),
        "library": source["build"]["library"],
        "field_sha256": old.trace.FIELD_SHA,
        "archive_sha256": old.trace.ARCHIVE_SHA,
        "seed": SEED,
    }
    path = ROOT / stage / "INPUTS.json"
    if path.exists() and json.loads(path.read_text())["binding"] != binding:
        raise ValueError("stage source/code changed; retain old generation and use new stage")
    record(
        path,
        {
            "binding": binding,
            "pointer": pointer,
            "axis": AXIS,
            "score_claim": False,
            "research_only": True,
            "argv": sys.argv,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "retention": "KEEP all payloads/checkpoints; no scratch, no deletion or movement",
            "free_bytes": shutil.disk_usage(ROOT.parent).free,
        },
    )
    return binding, library


def pack(frames, minimum):
    """Separately coded address and class/offset streams, all framing charged."""
    address, content = bytearray(), bytearray()
    for segments in frames:
        chosen = [s for s in segments if len(s[3]) >= minimum]
        address.extend(struct.pack("<I", len(chosen)))
        for key, vertex, directions, _positions, skip in chosen:
            a, b, normal = key
            content.append((a * 5 + b) * 3 + normal + 1)
            address.extend(struct.pack("<II", vertex, len(directions)))
            previous = 0
            for d, skipped in zip(directions, skip, strict=True):
                address.append((d - previous) % 4 + 4 * int(skipped))
                previous = d
    raw_a, raw_c = bytes(address), bytes(content)
    coded_a = brotli.compress(raw_a, quality=11, lgwin=22)
    coded_c = brotli.compress(raw_c, quality=11, lgwin=22)
    packet = HEADER.pack(b"BA3S", len(frames), old.H, old.W, len(coded_a), len(coded_c)) + coded_a + coded_c
    return {
        "address.raw": raw_a,
        "content.raw": raw_c,
        "address.br": coded_a,
        "content.br": coded_c,
        "segments": packet,
    }


def unpack(packet):
    if len(packet) < HEADER.size:
        raise ValueError("truncated split packet")
    magic, n, h, w, alen, clen = HEADER.unpack_from(packet)
    if magic != b"BA3S" or not 1 <= n <= 600 or (h, w) != (old.H, old.W) or HEADER.size + alen + clen != len(packet):
        raise ValueError("invalid split packet header")
    address = brotli.decompress(packet[HEADER.size : HEADER.size + alen])
    content = brotli.decompress(packet[HEADER.size + alen :])
    offset, ci, frames = 0, 0, []
    for _frame in range(n):
        (count,) = struct.unpack_from("<I", address, offset)
        offset += 4
        supplied = {}
        for _segment in range(count):
            vertex, length = struct.unpack_from("<II", address, offset)
            offset += 8
            if not length or offset + length > len(address) or ci >= len(content):
                raise ValueError("truncated segment")
            pair, normal = divmod(content[ci], 3)
            a, b = divmod(pair, 5)
            normal -= 1
            ci += 1
            if a >= 5 or a == b:
                raise ValueError("invalid segment class pair")
            d = 0
            for value in address[offset : offset + length]:
                if value > 7:
                    raise ValueError("invalid turn/skip code")
                d = (d + value % 4) % 4
                y, x = divmod(vertex, old.W + 1)
                dy, dx = old.DIRECTIONS[d]
                if not (0 <= y <= old.H and 0 <= x <= old.W and 0 <= y + dy <= old.H and 0 <= x + dx <= old.W):
                    raise ValueError("out-of-image crack")
                py, px = old.sample_edge(y, x, d, normal)
                if not (0 <= py < old.H and 0 <= px < old.W):
                    raise ValueError("out-of-image normal")
                if value < 4:
                    pos = py * old.W + px
                    if pos in supplied:
                        raise ValueError("duplicate supplied token")
                    supplied[pos] = b if normal == -1 else a
                vertex = joint.end_vertex(vertex, d)
            offset += length
        frames.append(supplied)
    if offset != len(address) or ci != len(content):
        raise ValueError("trailing split packet data")
    return frames


def geometry(orientation, gap, stop):
    stage = f"geometry_o{orientation}_k{gap}"
    binding, _ = bind(stage)
    for frame in range(stop):
        path = ROOT / stage / f"frame_{frame:04d}.json"
        if path.exists():
            if json.loads(path.read_text())["binding"] != binding:
                raise ValueError("geometry binding changed")
            continue
        data = old.load_frame(frame)
        segments, stats = joint.joint_segments(data["tokens"], data["best"], data["miss_pos"], orientation, gap)
        # Retain the first result before computing its independent solver twin.
        record(path.with_suffix(".primary.json"), {"segments": segments, "stats": stats})
        repeat, repeat_stats = joint.joint_segments(data["tokens"], data["best"], data["miss_pos"], orientation, gap)
        record(path.with_suffix(".repeat.json"), {"segments": repeat, "stats": repeat_stats})
        if segments != repeat or stats != repeat_stats:
            raise ValueError("joint solve twin differs")
        payload = pack([segments], 1)
        for name, value in payload.items():
            blob(ROOT / stage / "frame_packets" / f"{frame:04d}.{name}", value)
        expected = {p: int(data["tokens"].reshape(-1)[p]) for s in segments for p in s[3]}
        if unpack(payload["segments"])[0] != expected:
            raise ValueError("joint geometry packet parseback failed")
        if len(payload["address.raw"]) != stats["raw_address_bytes"]:
            raise ValueError("real raw serialization differs from MILP objective")
        record(path, {"binding": binding, "segments": segments, "stats": stats, "twin_identical": True})
        print(json.dumps({"stage": stage, "frame": frame + 1, **stats}), flush=True)


def decompose():
    binding, _ = bind("decompose")
    source = old.ROOT / "price_o1_min4"
    frames = []
    for frame in range(600):
        segments = json.loads((old.ROOT / "geometry_o1" / f"frame_{frame:04d}.json").read_text())["segments"]
        frames.append([[*s, [False] * len(s[2])] for s in segments])
    original_frames = [[s[:4] for s in segments] for segments in frames]
    primary, repeat = old.pack_segments(original_frames, 4), old.pack_segments(original_frames, 4)
    for tag, (raw, coded) in zip(("primary", "repeat"), (primary, repeat), strict=True):
        blob(ROOT / "decompose" / f"{tag}.old.raw", raw)
        blob(ROOT / "decompose" / f"{tag}.old.segments", coded)
    if primary != repeat or primary[1] != (source / "primary.segments").read_bytes():
        raise ValueError("bnd2 best segment reproduction failed")
    payloads = [pack(frames, 4), pack(frames, 4)]
    for tag, payload in zip(("primary", "repeat"), payloads, strict=True):
        for name, value in payload.items():
            blob(ROOT / "decompose" / f"{tag}.{name}", value)
    if payloads[0] != payloads[1] or unpack(payloads[0]["segments"]) != old.unpack_segments(primary[1]):
        raise ValueError("split twin/parseback mismatch")
    payload = payloads[0]
    raw = primary[0]
    _n, _h, _w, *sizes = struct.unpack_from("<HHH5I", raw, 4)
    cursor, sections = 30, []
    for size in sizes:
        sections.append(raw[cursor : cursor + size])
        cursor += size
    # Preserve bnd2's transforms, separating types from address streams.
    old_address = b"".join(sections[i] for i in (0, 2, 3, 4))
    for tag in ("primary", "repeat"):
        blob(ROOT / "decompose" / f"{tag}.old_address.raw", old_address)
        blob(ROOT / "decompose" / f"{tag}.old_address.br", brotli.compress(old_address, quality=11, lgwin=22))
        blob(ROOT / "decompose" / f"{tag}.old_content.raw", sections[1])
        blob(ROOT / "decompose" / f"{tag}.old_content.br", brotli.compress(sections[1], quality=11, lgwin=22))
    return record(
        ROOT / "decompose/RESULT.json",
        {
            "binding": binding,
            "axis": AXIS,
            "score_claim": False,
            "verdict_scope": "FORMULATION",
            "original_total": 121200,
            "original_residual": 119187,
            "original_framing": 16,
            "original_shared_brotli": len(primary[1]) - 12,
            "original_segment_plus_length": len(primary[1]) + 4,
            "physical_address_content_split": "NOT_IDENTIFIABLE: shared Brotli stream",
            "split_address_bytes": len(payload["address.br"]),
            "split_content_bytes": len(payload["content.br"]),
            "split_framing_bytes": HEADER.size + 4,
            "split_total": 119187 + 4 + len(payload["segments"]),
            "old_transform_address": old.jg2.file_fact(ROOT / "decompose/primary.old_address.br"),
            "old_transform_content": old.jg2.file_fact(ROOT / "decompose/primary.old_content.br"),
            "supplied_tokens": sum(len(x) for x in unpack(payload["segments"])),
            "twin_identical": True,
        },
    )


def encode(stage, frames, minimum, oracle, stop):
    binding, library = bind(stage)
    work = ROOT / stage
    supplied = None
    if not oracle:
        payloads = [pack(frames, minimum), pack(frames, minimum)]
        for tag, payload in zip(("primary", "repeat"), payloads, strict=True):
            for name, value in payload.items():
                blob(work / f"{tag}.{name}", value)
        if payloads[0] != payloads[1]:
            raise ValueError("split packet twins differ")
        segment = payloads[0]["segments"]
        supplied = unpack(segment)
        if len(supplied) != 600:
            raise ValueError("not full n600 geometry")
    route = old.jg2.load_route_b()
    order = np.frombuffer((old.ROOT / "trace/group_order.i32").read_bytes(), dtype="<i4")
    state, start = None, 0
    latest = work / "LATEST.json"
    if latest.exists():
        receipt = json.loads(latest.read_text())
        if receipt["binding"] != binding or old.jg2.file_fact(Path(receipt["payload"]["path"])) != receipt["payload"]:
            raise ValueError("encode state custody changed")
        with np.load(receipt["payload"]["path"], allow_pickle=False) as data:
            state = {k: data[k] for k in data.files}
        start = int(state["frame"][0])
    encoders = [
        route.NativeRc64Encoder(library, None if state is None else state[k].tobytes())
        for k in ("encoder0", "encoder1")
    ]
    for frame in range(start, stop):
        data = old.load_frame(frame)
        truth = data["tokens"].reshape(-1)
        positions = data["miss_pos"] if oracle else np.array(sorted(supplied[frame]), dtype=np.int32)
        if not oracle and (
            any(truth[p] != symbol for p, symbol in supplied[frame].items())
            or not np.isin(positions, data["miss_pos"]).all()
        ):
            raise ValueError("supplied token is not a true coder miss")
        # Oracle support AND true symbol are supplied free. All actual symbols
        # still enter the causal model; the cached rows are therefore unchanged.
        if oracle:
            blob(work / "oracle_support" / f"frame_{frame:04d}.i32", positions.astype("<i4").tobytes())
            blob(work / "oracle_content" / f"frame_{frame:04d}.u8", truth[positions].tobytes())
        mask = np.ones(old.H * old.W, dtype=bool)
        mask[positions] = False
        selected = order[mask[order]]
        for encoder in encoders:
            encoder.encode(truth[selected].astype(np.int32), data["rows"][selected])
        if (frame + 1) % 20 == 0 or frame + 1 == stop:
            arrays = {
                "frame": np.array([frame + 1]),
                **{f"encoder{i}": np.frombuffer(enc.snapshot(), dtype=np.uint8) for i, enc in enumerate(encoders)},
            }
            fact = checkpoint(work / "checkpoints" / f"encode_{frame + 1:04d}.npz", arrays)
            record(latest, {"binding": binding, "payload": fact})
            print(json.dumps({"stage": stage, "frame": frame + 1}), flush=True)
    if stop != 600:
        return record(
            work / f"PARTIAL_{stop:04d}.json", {"binding": binding, "frames": stop, "prices_admissible": False}
        )
    residuals = []
    for tag, encoder in zip(("primary", "repeat"), encoders, strict=True):
        blob(work / f"{tag}.native.envelope", encoder.finish())
        size = int(encoder.library.rc64_encoder_size(encoder.context))
        raw = ctypes.string_at(encoder.library.rc64_encoder_data(encoder.context), size)
        blob(work / f"{tag}.residual.raw", raw)
        value = b"R6D1" + raw + b"\0"
        blob(work / f"{tag}.residual.envelope", value)
        residuals.append(value)
    if residuals[0] != residuals[1]:
        raise ValueError("native twin differs")
    decoder = route.NativeRc64Decoder(library, residuals[0])
    decode_latest = work / "DECODE_LATEST.json"
    decode_start = 0
    if decode_latest.exists():
        receipt = json.loads(decode_latest.read_text())
        if receipt["binding"] != binding or old.jg2.file_fact(Path(receipt["payload"]["path"])) != receipt["payload"]:
            raise ValueError("decode state custody changed")
        decoder = route.NativeRc64Decoder(library, Path(receipt["payload"]["path"]).read_bytes())
        decode_start = receipt["frame"]
    for frame in range(decode_start, 600):
        data = old.load_frame(frame)
        truth = data["tokens"].reshape(-1)
        support = {int(p): int(truth[p]) for p in data["miss_pos"]} if oracle else supplied[frame]
        restored, mask = np.zeros_like(truth), np.ones(old.H * old.W, dtype=bool)
        for pos, symbol in support.items():
            restored[pos], mask[pos] = symbol, False
        positions = order[mask[order]]
        restored[positions] = decoder.decode(None, data["rows"][positions]).astype(np.uint8)
        blob(work / "decoded" / f"frame_{frame:04d}.u8", restored.tobytes())
        if not np.array_equal(restored, truth):
            raise ValueError("full-field cached-row parseback failed")
        if (frame + 1) % 20 == 0:
            fact = blob(work / "checkpoints" / f"decode_{frame + 1:04d}.bin", decoder.get_compressed().tobytes())
            record(decode_latest, {"binding": binding, "payload": fact, "frame": frame + 1})
    digest = hashlib.sha256()
    for frame in range(600):
        digest.update((work / "decoded" / f"frame_{frame:04d}.u8").read_bytes())
    if digest.hexdigest() != old.trace.FIELD_SHA:
        raise ValueError("decoded field sha mismatch")
    result = {
        "binding": binding,
        "axis": AXIS,
        "score_claim": False,
        "research_only": True,
        "frames": 600,
        "symbols": 117964800,
        "baseline_envelope_bytes": 119784,
        "residual_bytes": len(residuals[0]),
        "twin_identical": True,
        "decoded_field_sha256": digest.hexdigest(),
        "cached_rows_parseback": True,
        "fresh_causal_decoder": False,
        "resumed_from_frame": start,
        "oracle": oracle,
        "verdict_scope": "FORMULATION",
    }
    if oracle:
        result.update(
            displaceable_bytes=119784 - len(residuals[0]),
            oracle_supplied_tokens=235044,
            boundary="Free true symbols AND locations; same causal trajectory; bound only for this fixed TC1 remainder, not all codecs",
        )
    else:
        for tag, residual in zip(("primary", "repeat"), residuals, strict=True):
            value = b"BND3" + struct.pack("<I", len(segment)) + segment + residual[4:]
            blob(work / f"{tag}.replacement.envelope", value)
            member = old.jg2.read_archive_member(old.ROOT / "runtime/archive.zip")
            weights = (old.trace.ROOT / "retained/tc1_weights.bin").read_bytes()
            old_rider = b"TC1M\x01" + weights + (old.trace.ROOT / "retained/shipped_token_stream.rc64").read_bytes()
            if not member.endswith(old_rider):
                raise ValueError("source rider mismatch")
            new_member = member[: -len(old_rider)] + b"BND3\x01" + weights + value[4:-1]
            blob(work / f"{tag}.archive_member", new_member)
            destination = work / ("archive.zip" if tag == "primary" else "archive.repeat.zip")
            if not destination.exists():
                old.jg2.pack_archive(new_member, destination)
            if old.jg2.read_archive_member(destination) != new_member:
                raise ValueError("ZIP parseback failed")
        if (work / "archive.zip").read_bytes() != (work / "archive.repeat.zip").read_bytes():
            raise ValueError("ZIP twins differ")
        total = 4 + len(segment) + len(residuals[0])
        if (work / "archive.zip").stat().st_size - 180388 != total - 119784:
            raise ValueError("ZIP/envelope accounting mismatch")
        result.update(
            address_bytes=len(payloads[0]["address.br"]),
            content_bytes=len(payloads[0]["content.br"]),
            framing_bytes=HEADER.size + 4,
            total_envelope_bytes=total,
            delta_bytes=total - 119784,
            archive=old.jg2.file_fact(work / "archive.zip"),
            minimum=minimum,
            supplied_tokens=sum(len(x) for x in supplied),
            draw_gate=total <= 114784,
        )
    return record(work / "RESULT.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("decompose", "geometry", "oracle", "price"))
    parser.add_argument("--resume-from", required=True, type=Path)
    parser.add_argument("--orientation", type=int, choices=(0, 1), default=1)
    parser.add_argument("--gap", type=int, choices=(0, 1, 2, 4), default=0)
    parser.add_argument("--minimum", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--stop-after", type=int, default=600)
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT.resolve() or not 1 <= args.stop_after <= 600:
        raise ValueError("invalid resume root/frame count")
    np.random.seed(SEED)
    if args.stage == "decompose":
        result = decompose()
    elif args.stage == "geometry":
        result = geometry(args.orientation, args.gap, args.stop_after)
    elif args.stage == "oracle":
        result = encode("oracle", None, 1, True, args.stop_after)
    else:
        frames = [
            json.loads((ROOT / f"geometry_o{args.orientation}_k{args.gap}" / f"frame_{f:04d}.json").read_text())[
                "segments"
            ]
            for f in range(600)
        ]
        result = encode(
            f"price_o{args.orientation}_k{args.gap}_m{args.minimum}", frames, args.minimum, False, args.stop_after
        )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
