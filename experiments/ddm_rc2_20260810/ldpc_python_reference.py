#!/usr/bin/env python3
"""Python reference decoder for the ddm_rc2 Rust LDPC token packet.

The reference deliberately mirrors the finite wire semantics: deterministic
sparse graph construction, normalized min-sum BP, the miss-class frequency
quantizer, and the arithmetic decoder.  It writes the decoded payload and a
parity receipt atomically; an optional retained bit-flip control must not
reproduce the source object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"RC2LDPC1"
FRAME_SYMBOLS = 384 * 512
GROUPS_PER_FRAME = (1 + 2) * 64 - 2
ARITH_TOTAL = 32768
HALF = 0x80000000
QUARTER = 0x40000000
THREE_QUARTERS = 0xC0000000
FULL = 0xFFFFFFFF
AXIS = "[macOS-CPU advisory, scorer-free]"


@dataclass(frozen=True)
class Packet:
    symbol_count: int
    group_count: int
    alpha_milli: int
    degree: int
    iterations: int
    group_offset: int
    flag_bits: int
    hit_bits: int
    miss_bits: int
    flags: bytes
    hit_payload: bytes
    miss_payload: bytes


class BitReader:
    def __init__(self, payload: bytes, limit: int) -> None:
        self.payload = payload
        self.limit = limit
        self.position = 0

    def read(self, *, pad_zero: bool = False) -> bool:
        if self.position >= self.limit:
            if pad_zero:
                return False
            raise RuntimeError("bitstream exhausted")
        byte = self.payload[self.position // 8]
        bit = bool((byte >> (7 - (self.position % 8))) & 1)
        self.position += 1
        return bit

    def read_many(self, count: int) -> list[bool]:
        return [self.read() for _ in range(count)]


class ArithmeticDecoder:
    def __init__(self, payload: bytes, bit_count: int) -> None:
        self.bits = BitReader(payload, bit_count)
        self.low = 0
        self.high = FULL
        self.value = 0
        for _ in range(32):
            self.value = (self.value << 1) | int(self.bits.read(pad_zero=True))

    def decode(self, cumulative: tuple[int, int, int, int, int]) -> int:
        total = cumulative[4]
        interval = self.high - self.low + 1
        scaled = ((self.value - self.low + 1) * total - 1) // interval
        symbol = next(index for index in range(4) if scaled < cumulative[index + 1])
        self.high = self.low + interval * cumulative[symbol + 1] // total - 1
        self.low += interval * cumulative[symbol] // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.value -= HALF
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTERS:
                self.value -= QUARTER
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low = (self.low << 1) & FULL
            self.high = ((self.high << 1) | 1) & FULL
            self.value = ((self.value << 1) & FULL) | int(self.bits.read(pad_zero=True))
        return symbol


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def parse_packet(payload: bytes) -> Packet:
    if payload[:8] != MAGIC:
        raise RuntimeError("LDPC packet magic mismatch")
    cursor = 8

    def read(fmt: str) -> int:
        nonlocal cursor
        value = struct.unpack_from(fmt, payload, cursor)[0]
        cursor += struct.calcsize(fmt)
        return int(value)

    symbol_count = read("<I")
    group_count = read("<I")
    alpha_milli = read("<H")
    degree = read("<B")
    iterations = read("<B")
    group_offset = read("<Q")
    flag_bits = read("<Q")
    hit_bits = read("<Q")
    miss_bits = read("<Q")
    flag_bytes = read("<I")
    hit_bytes = read("<I")
    miss_bytes = read("<I")
    flags = payload[cursor : cursor + flag_bytes]
    cursor += flag_bytes
    hit_payload = payload[cursor : cursor + hit_bytes]
    cursor += hit_bytes
    miss_payload = payload[cursor : cursor + miss_bytes]
    cursor += miss_bytes
    if cursor != len(payload):
        raise RuntimeError("packet has trailing or truncated bytes")
    return Packet(
        symbol_count,
        group_count,
        alpha_milli,
        degree,
        iterations,
        group_offset,
        flag_bits,
        hit_bits,
        miss_bits,
        flags,
        hit_payload,
        miss_payload,
    )


def splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def build_graph(n: int, m: int, degree: int, seed: int) -> tuple[list[int], list[list[int]], list[list[int]]]:
    edge_var: list[int] = []
    var_edges = [[] for _ in range(n)]
    check_edges = [[] for _ in range(m)]
    if m == 0:
        return edge_var, var_edges, check_edges
    for variable in range(n):
        selected: list[int] = []
        attempt = 0
        while len(selected) < min(degree, m):
            mixed = splitmix64(
                seed
                ^ ((variable * 0xD6E8FEB86659FD93) & ((1 << 64) - 1))
                ^ ((attempt * 0xA0761D6478BD642F) & ((1 << 64) - 1))
            )
            check = mixed % m
            if check not in selected:
                selected.append(check)
            attempt += 1
        for check in selected:
            edge = len(edge_var)
            edge_var.append(variable)
            var_edges[variable].append(edge)
            check_edges[check].append(edge)
    return edge_var, var_edges, check_edges


def syndrome_matches(edge_var: list[int], check_edges: list[list[int]], bits: list[bool], target: list[bool]) -> bool:
    return all(
        bool(sum(bits[edge_var[edge]] for edge in edges) & 1) == expected
        for edges, expected in zip(check_edges, target, strict=True)
    )


def bp_decode(
    edge_var: list[int],
    var_edges: list[list[int]],
    check_edges: list[list[int]],
    priors: list[float],
    target: list[bool],
    iterations: int,
) -> list[bool]:
    q = [priors[variable] for variable in edge_var]
    r = [0.0] * len(q)
    hard = [value < 0.0 for value in priors]
    if syndrome_matches(edge_var, check_edges, hard, target):
        return hard
    for _ in range(iterations):
        for check, edges in enumerate(check_edges):
            for edge in edges:
                sign = -1.0 if target[check] else 1.0
                minimum = 20.0
                for other in edges:
                    if other == edge:
                        continue
                    sign *= -1.0 if q[other] < 0.0 else 1.0
                    minimum = min(minimum, abs(q[other]))
                r[edge] = 0.8 * sign * minimum
        for variable, edges in enumerate(var_edges):
            posterior = priors[variable] + sum(r[edge] for edge in edges)
            hard[variable] = posterior < 0.0
            for edge in edges:
                q[edge] = max(-20.0, min(20.0, posterior - r[edge]))
        if syndrome_matches(edge_var, check_edges, hard, target):
            return hard
    raise RuntimeError("Python BP failed for a non-fallback group")


def group_sizes() -> list[int]:
    local = [0] * GROUPS_PER_FRAME
    for row in range(64):
        for column in range(64):
            local[column + 2 * row] += 1
    return [count * 48 for count in local]


def probabilities(codes: np.ndarray) -> tuple[list[float], int]:
    maximum = max(int(value) for value in codes) / 8.0
    weights = [math.exp(int(value) / 8.0 - maximum) for value in codes]
    total = sum(weights)
    probability = [value / total for value in weights]
    # Match the PR130 NumPy/Torch argmax contract: first equal maximum wins.
    top = max(range(5), key=probability.__getitem__)
    return probability, top


def binary_entropy(probability: float) -> float:
    p = max(1e-15, min(1.0 - 1e-15, probability))
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def miss_cumulative(probability: list[float], top: int) -> tuple[list[int], tuple[int, int, int, int, int]]:
    alternatives = [class_id for class_id in range(5) if class_id != top]
    total_probability = sum(probability[class_id] for class_id in alternatives)
    exact = [probability[class_id] / total_probability * (ARITH_TOTAL - 4) for class_id in alternatives]
    frequencies = [math.floor(value) + 1 for value in exact]
    fractions = [value - math.floor(value) for value in exact]
    allocated = sum(frequencies)
    while allocated < ARITH_TOTAL:
        index = max(range(4), key=lambda candidate: (fractions[candidate], candidate))
        frequencies[index] += 1
        fractions[index] = -1.0
        allocated += 1
    while allocated > ARITH_TOTAL:
        index = max(
            (candidate for candidate in range(4) if frequencies[candidate] > 1),
            key=lambda candidate: (frequencies[candidate], candidate),
        )
        frequencies[index] -= 1
        allocated -= 1
    cumulative = [0]
    for frequency in frequencies:
        cumulative.append(cumulative[-1] + int(frequency))
    return alternatives, tuple(cumulative)  # type: ignore[return-value]


def decode_reference(codes: np.ndarray, packet: Packet) -> bytes:
    if codes.shape != (packet.symbol_count, 5) or packet.symbol_count % FRAME_SYMBOLS:
        raise RuntimeError("codes shape does not match packet")
    flags = BitReader(packet.flags, packet.flag_bits)
    hits = BitReader(packet.hit_payload, packet.hit_bits)
    misses = ArithmeticDecoder(packet.miss_payload, packet.miss_bits)
    sizes = group_sizes()
    output = bytearray()
    offset = 0
    for group_index in range(packet.group_count):
        n = sizes[group_index % GROUPS_PER_FRAME]
        group_codes = codes[offset : offset + n]
        probability_rows: list[list[float]] = []
        tops: list[int] = []
        priors: list[float] = []
        entropy = 0.0
        for row in group_codes:
            probability, top = probabilities(row)
            p_hit = max(1e-15, min(1.0 - 1e-15, float(probability[top])))
            probability_rows.append(probability)
            tops.append(top)
            priors.append(max(-20.0, min(20.0, math.log((1.0 - p_hit) / p_hit))))
            entropy += binary_entropy(p_hit)
        m = min(n, math.ceil(entropy * packet.alpha_milli / 1000.0))
        if flags.read():
            decoded_hits = hits.read_many(n)
        else:
            target = hits.read_many(m)
            edge_var, var_edges, check_edges = build_graph(
                n,
                m,
                packet.degree,
                splitmix64(packet.group_offset + group_index),
            )
            decoded_hits = bp_decode(
                edge_var,
                var_edges,
                check_edges,
                priors,
                target,
                packet.iterations,
            )
        for index in range(n):
            if decoded_hits[index]:
                output.append(tops[index])
            else:
                alternatives, cumulative = miss_cumulative(probability_rows[index], tops[index])
                output.append(alternatives[misses.decode(cumulative)])
        offset += n
    if offset != packet.symbol_count or flags.position != packet.flag_bits or hits.position != packet.hit_bits:
        raise RuntimeError("reference decoder did not consume declared symbols/bits")
    return bytes(output)


def run(args: argparse.Namespace) -> dict[str, Any]:
    packet_bytes = args.packet.read_bytes()
    packet = parse_packet(packet_bytes)
    codes = np.load(args.codes, mmap_mode="r", allow_pickle=False)
    started = time.perf_counter()
    decoded = decode_reference(codes, packet)
    decode_s = time.perf_counter() - started
    atomic_bytes(args.output, decoded)
    source = np.load(args.symbols, mmap_mode="r", allow_pickle=False)
    source_bytes = np.asarray(source).tobytes(order="C")
    exact = decoded == source_bytes
    rust_exact = args.rust_decoded.read_bytes() == decoded
    if not exact or not rust_exact:
        raise RuntimeError("Python reference, Rust decode, and source are not byte-identical")
    result: dict[str, Any] = {
        "axis": AXIS,
        "score_claim": False,
        "packet": {"path": str(args.packet), "bytes": len(packet_bytes), "sha256": sha256_bytes(packet_bytes)},
        "codes": {"path": str(args.codes), "sha256": sha256_file(args.codes)},
        "symbols": {"path": str(args.symbols), "sha256": sha256_file(args.symbols)},
        "python_decoded": {"path": str(args.output), "bytes": len(decoded), "sha256": sha256_bytes(decoded)},
        "rust_decoded": {
            "path": str(args.rust_decoded),
            "bytes": args.rust_decoded.stat().st_size,
            "sha256": sha256_file(args.rust_decoded),
        },
        "source_payload_sha256": sha256_bytes(source_bytes),
        "python_rust_source_byte_identical": True,
        "python_decode_seconds": decode_s,
    }
    if args.negative_control:
        mutated = bytearray(packet_bytes)
        mutation_index = len(MAGIC) + 56
        if mutation_index >= len(mutated):
            raise RuntimeError("packet too small for retained bit-flip control")
        mutated[mutation_index] ^= 1
        mutated_path = args.output.with_name("packet.bitflip.bin")
        atomic_bytes(mutated_path, bytes(mutated))
        negative_output = args.output.with_name("python_decoded.bitflip.bin")
        negative_error = None
        try:
            negative_decoded = decode_reference(codes, parse_packet(bytes(mutated)))
            atomic_bytes(negative_output, negative_decoded)
            negative_equal = negative_decoded == source_bytes
        except Exception as error:  # A loud decode failure is a valid negative-control outcome.
            negative_equal = False
            negative_error = f"{type(error).__name__}: {error}"
        if negative_equal:
            raise RuntimeError("bit-flip negative control reproduced the source")
        result["negative_control"] = {
            "mutation": f"xor bit 0 at byte {mutation_index}",
            "packet": {
                "path": str(mutated_path),
                "bytes": mutated_path.stat().st_size,
                "sha256": sha256_file(mutated_path),
            },
            "decoded_path": str(negative_output) if negative_output.exists() else None,
            "decoded_sha256": sha256_file(negative_output) if negative_output.exists() else None,
            "error": negative_error,
            "source_equality": False,
            "passed": True,
        }
    receipt = args.output.with_name("python_parity_receipt.json")
    atomic_json(receipt, result)
    result["receipt"] = {"path": str(receipt), "bytes": receipt.stat().st_size, "sha256": sha256_file(receipt)}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--symbols", type=Path, required=True)
    parser.add_argument("--rust-decoded", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-control", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
