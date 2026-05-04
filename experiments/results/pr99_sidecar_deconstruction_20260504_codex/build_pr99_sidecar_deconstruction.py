#!/usr/bin/env python3
"""Local PR99 latent-correction sidecar deconstruction and candidate builder.

This script is deliberately local-only: it does not submit jobs, mutate dispatch
claims, or make score claims. It inspects PR99's one-dimensional per-pair
latent-correction sidecar, then emits deterministic no-dispatch candidates that
preserve the decoded correction arrays exactly.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import sys
import zipfile
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

import brotli
import numpy as np


REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "experiments/results/pr99_sidecar_deconstruction_20260504_codex"
SOURCE_ARCHIVE = REPO / "experiments/results/leaderboard_intel_20260504_codex/pr99_archive.zip"
SOURCE_RUNTIME = REPO / "experiments/results/leaderboard_intel_20260504_codex/pr99_runtime"
SANITIZED_RUNTIME = (
    REPO
    / "experiments/results/final_packet_readiness_pr98_pr99_20260504_codex/runtime_snapshots/pr99_runtime"
)
UPSTREAM = REPO / "upstream"


def digest_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def digest_file(file_path: Path) -> str:
    h = sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_member(archive_path: Path) -> bytes:
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        if names != ["0.bin"]:
            raise RuntimeError(f"expected exactly ['0.bin'], got {names!r}")
        info = zf.getinfo("0.bin")
        if info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError(f"expected stored 0.bin, got compress_type={info.compress_type}")
        return zf.read("0.bin")


def split_body(body: bytes) -> list[bytes]:
    off = 0
    parts: list[bytes] = []
    for _ in range(4):
        if off + 4 > len(body):
            raise RuntimeError("truncated length prefix")
        length = struct.unpack_from("<I", body, off)[0]
        off += 4
        part = body[off : off + length]
        if len(part) != length:
            raise RuntimeError("truncated payload part")
        parts.append(part)
        off += length
    if off != len(body):
        raise RuntimeError(f"trailing body bytes: {off} vs {len(body)}")
    return parts


def join_body(parts: list[bytes]) -> bytes:
    out = bytearray()
    for part in parts:
        out += struct.pack("<I", len(part))
        out += part
    return bytes(out)


def write_zip(archive_path: Path, body: bytes, *, compress_type: int) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = compress_type
    info.create_system = 0
    info.external_attr = 0o100644 << 16
    if compress_type == zipfile.ZIP_DEFLATED:
        with zipfile.ZipFile(archive_path, "w", strict_timestamps=True) as zf:
            zf.writestr(info, body, compress_type=compress_type, compresslevel=9)
    else:
        with zipfile.ZipFile(archive_path, "w", strict_timestamps=True) as zf:
            zf.writestr(info, body, compress_type=compress_type)


def decode_current_sidecar(blob: bytes) -> tuple[np.ndarray, np.ndarray, bytes]:
    raw = brotli.decompress(blob)
    n = struct.unpack_from("<H", raw, 0)[0]
    arr = np.frombuffer(raw[2 : 2 + 2 * n], dtype=np.uint8).reshape(n, 2)
    if len(raw) != 2 + 2 * n:
        raise RuntimeError(f"sidecar raw trailing bytes: {len(raw)} vs {2 + 2 * n}")
    return arr[:, 0].copy(), arr[:, 1].view(np.int8).copy(), raw


def pack_bits(values: list[int], width: int) -> bytes:
    acc = 0
    bits = 0
    out = bytearray()
    mask = (1 << width) - 1
    for value in values:
        if value < 0 or value > mask:
            raise ValueError(value)
        acc |= value << bits
        bits += width
        while bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            bits -= 8
    if bits:
        out.append(acc & 0xFF)
    return bytes(out)


def unpack_bits(data: bytes, count: int, width: int) -> list[int]:
    acc = 0
    bits = 0
    idx = 0
    out: list[int] = []
    mask = (1 << width) - 1
    for _ in range(count):
        while bits < width:
            if idx >= len(data):
                raise RuntimeError("bitstream underflow")
            acc |= data[idx] << bits
            idx += 1
            bits += 8
        out.append(acc & mask)
        acc >>= width
        bits -= width
    return out


def make_split_stream(dim: np.ndarray, delta_q: np.ndarray) -> bytes:
    dim_b = brotli.compress(dim.astype(np.uint8).tobytes(), quality=11)
    delta_b = brotli.compress(delta_q.astype(np.int8).view(np.uint8).tobytes(), quality=11)
    return b"LC2S" + struct.pack("<HH", len(dim), len(dim_b)) + dim_b + delta_b


def decode_split_stream(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    n, dim_len = struct.unpack_from("<HH", blob, 4)
    dim_b = blob[8 : 8 + dim_len]
    delta_b = blob[8 + dim_len :]
    dim = np.frombuffer(brotli.decompress(dim_b), dtype=np.uint8).copy()
    delta = np.frombuffer(brotli.decompress(delta_b), dtype=np.uint8).view(np.int8).copy()
    if len(dim) != n or len(delta) != n:
        raise RuntimeError("split stream length mismatch")
    return dim, delta


def make_bitpack_stream(dim: np.ndarray, delta_q: np.ndarray) -> bytes:
    corrected = dim != 255
    bitmap = np.packbits(corrected.astype(np.uint8), bitorder="little").tobytes()
    corrected_dims = [int(x) for x in dim[corrected]]
    dim_bits = pack_bits(corrected_dims, 5)
    delta = delta_q.astype(np.int8).view(np.uint8).tobytes()
    raw = struct.pack("<HHH", len(dim), len(bitmap), len(dim_bits)) + bitmap + dim_bits + delta
    return b"LCBP" + brotli.compress(raw, quality=11)


def decode_bitpack_stream(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    raw = brotli.decompress(blob[4:])
    n, bitmap_len, dim_bits_len = struct.unpack_from("<HHH", raw, 0)
    cursor = 6
    bitmap = raw[cursor : cursor + bitmap_len]
    cursor += bitmap_len
    dim_bits = raw[cursor : cursor + dim_bits_len]
    cursor += dim_bits_len
    delta = np.frombuffer(raw[cursor : cursor + n], dtype=np.uint8).view(np.int8).copy()
    corrected = np.unpackbits(np.frombuffer(bitmap, dtype=np.uint8), bitorder="little")[:n].astype(bool)
    dims = unpack_bits(dim_bits, int(corrected.sum()), 5)
    dim = np.full(n, 255, dtype=np.uint8)
    dim[corrected] = np.array(dims, dtype=np.uint8)
    if len(delta) != n:
        raise RuntimeError("bitpack delta length mismatch")
    return dim, delta


def make_sparse_stream(dim: np.ndarray, delta_q: np.ndarray) -> bytes:
    corrected_idx = np.nonzero(dim != 255)[0].astype(np.uint16)
    corrected_dim = dim[corrected_idx].astype(np.uint8)
    corrected_delta = delta_q[corrected_idx].astype(np.int8).view(np.uint8)
    raw = (
        struct.pack("<HH", len(dim), len(corrected_idx))
        + corrected_idx.tobytes()
        + corrected_dim.tobytes()
        + corrected_delta.tobytes()
    )
    return b"LCSP" + brotli.compress(raw, quality=11)


def decode_sparse_stream(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    raw = brotli.decompress(blob[4:])
    n, k = struct.unpack_from("<HH", raw, 0)
    cursor = 4
    idx = np.frombuffer(raw[cursor : cursor + 2 * k], dtype=np.uint16)
    cursor += 2 * k
    vals_dim = np.frombuffer(raw[cursor : cursor + k], dtype=np.uint8)
    cursor += k
    vals_delta = np.frombuffer(raw[cursor : cursor + k], dtype=np.uint8).view(np.int8)
    dim = np.full(n, 255, dtype=np.uint8)
    delta = np.zeros(n, dtype=np.int8)
    dim[idx] = vals_dim
    delta[idx] = vals_delta
    return dim, delta


def best_current_wire(raw_sidecar: bytes) -> dict[str, Any]:
    records = []
    for quality in range(12):
        for lgwin in range(10, 25):
            try:
                blob = brotli.compress(raw_sidecar, quality=quality, lgwin=lgwin)
            except brotli.error:
                continue
            if brotli.decompress(blob) != raw_sidecar:
                raise RuntimeError("brotli roundtrip mismatch")
            records.append({"quality": quality, "lgwin": lgwin, "bytes": len(blob), "sha256": digest_bytes(blob)})
    records.sort(key=lambda r: (r["bytes"], r["quality"], r["lgwin"]))
    return {"best": records[0], "records": records[:24]}


SIDE_CAR_MULTI_CODEC = '''"""Latent-correction sidecar for hnerv_repack_latent.

Supports PR99's original brotli'd two-byte-per-pair wire format plus local
lossless candidate formats used for no-dispatch byte screens.
"""
import struct
import numpy as np

DELTA_SCALE = 0.01


def _unpack_bits(data, count, width):
    acc = 0
    bits = 0
    idx = 0
    out = []
    mask = (1 << width) - 1
    for _ in range(count):
        while bits < width:
            if idx >= len(data):
                raise RuntimeError("bitstream underflow")
            acc |= data[idx] << bits
            idx += 1
            bits += 8
        out.append(acc & mask)
        acc >>= width
        bits -= width
    return out


def decode_corrections(blob):
    import brotli
    if blob.startswith(b"LC2S"):
        n, dim_len = struct.unpack_from("<HH", blob, 4)
        dim_b = blob[8:8 + dim_len]
        delta_b = blob[8 + dim_len:]
        dim = np.frombuffer(brotli.decompress(dim_b), dtype=np.uint8).copy()
        delta_q = np.frombuffer(brotli.decompress(delta_b), dtype=np.uint8).view(np.int8).copy()
        if len(dim) != n or len(delta_q) != n:
            raise RuntimeError("LC2S length mismatch")
        return dim, delta_q
    if blob.startswith(b"LCBP"):
        raw = brotli.decompress(blob[4:])
        n, bitmap_len, dim_bits_len = struct.unpack_from("<HHH", raw, 0)
        cursor = 6
        bitmap = raw[cursor:cursor + bitmap_len]
        cursor += bitmap_len
        dim_bits = raw[cursor:cursor + dim_bits_len]
        cursor += dim_bits_len
        delta_q = np.frombuffer(raw[cursor:cursor + n], dtype=np.uint8).view(np.int8).copy()
        corrected = np.unpackbits(np.frombuffer(bitmap, dtype=np.uint8), bitorder="little")[:n].astype(bool)
        dims = _unpack_bits(dim_bits, int(corrected.sum()), 5)
        dim = np.full(n, 255, dtype=np.uint8)
        dim[corrected] = np.array(dims, dtype=np.uint8)
        if len(delta_q) != n:
            raise RuntimeError("LCBP delta length mismatch")
        return dim, delta_q
    if blob.startswith(b"LCSP"):
        raw = brotli.decompress(blob[4:])
        n, k = struct.unpack_from("<HH", raw, 0)
        cursor = 4
        idx = np.frombuffer(raw[cursor:cursor + 2 * k], dtype=np.uint16)
        cursor += 2 * k
        vals_dim = np.frombuffer(raw[cursor:cursor + k], dtype=np.uint8)
        cursor += k
        vals_delta = np.frombuffer(raw[cursor:cursor + k], dtype=np.uint8).view(np.int8)
        dim = np.full(n, 255, dtype=np.uint8)
        delta_q = np.zeros(n, dtype=np.int8)
        dim[idx] = vals_dim
        delta_q[idx] = vals_delta
        return dim, delta_q
    raw = brotli.decompress(blob)
    n = struct.unpack_from("<H", raw, 0)[0]
    arr = np.frombuffer(raw[2:2 + 2*n], dtype=np.uint8).reshape(n, 2)
    dim = arr[:, 0]
    delta_q = arr[:, 1].view(np.int8)
    return dim, delta_q


def encode_corrections(out_dim, out_delta_q):
    import brotli
    n = len(out_dim)
    assert len(out_delta_q) == n
    dim_packed = np.where(out_delta_q == 0, 255, out_dim).astype(np.uint8)
    payload = struct.pack("<H", n) + np.stack([dim_packed, out_delta_q.astype(np.int8).view(np.uint8)], axis=1).tobytes()
    return brotli.compress(payload, quality=11)


def apply_corrections(latents_tensor, dim_arr, delta_q_arr, scale=DELTA_SCALE):
    n = latents_tensor.shape[0]
    for p in range(n):
        d = int(dim_arr[p])
        if d == 255:
            continue
        latents_tensor[p, d] = latents_tensor[p, d] + float(delta_q_arr[p]) * scale
    return latents_tensor
'''


def copy_runtime_variant(name: str, *, multi_codec: bool) -> Path:
    runtime_root = OUT / "runtime_variants" / name
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    shutil.copytree(SANITIZED_RUNTIME, runtime_root)
    if multi_codec:
        (runtime_root / "sidecar.py").write_text(SIDE_CAR_MULTI_CODEC, encoding="utf-8")
    for py_file in sorted(runtime_root.glob("*.py")):
        compile(py_file.read_text(encoding="utf-8"), str(py_file), "exec")
    subprocess.run(["bash", "-n", str(runtime_root / "inflate.sh")], check=True)
    return runtime_root


def emit_plan(candidate: dict[str, Any], runtime_root: Path, archive_path: Path) -> dict[str, Any]:
    job = f"exact_eval_pr99_sidecar_{candidate['id']}_t4_YYYYMMDDTHHMMSSZ"
    lane = f"pr99_sidecar_{candidate['id']}_t4_screen"
    command = [
        ".venv/bin/python",
        "scripts/launch_lightning_batch_job.py",
        "exact-eval",
        "--state-path",
        ".omx/state/lightning_batch_jobs.json",
        "--job-name",
        job,
        "--archive",
        str(archive_path),
        "--inflate-sh",
        str(runtime_root / "inflate.sh"),
        "--machine",
        "T4_SMALL",
        "--adjudicate",
        "--expected-archive-sha256",
        candidate["archive_sha256"],
        "--expected-archive-size-bytes",
        str(candidate["archive_bytes"]),
    ]
    return {
        "schema": "pr99_sidecar_no_dispatch_exact_eval_plan_v1",
        "dispatch_performed": False,
        "claim_required_before_dispatch": True,
        "suggested_lane_id": lane,
        "suggested_job_name": job,
        "archive": str(archive_path.relative_to(REPO)),
        "archive_bytes": candidate["archive_bytes"],
        "archive_sha256": candidate["archive_sha256"],
        "runtime_root": str(runtime_root.relative_to(REPO)),
        "runtime_tree_sha256": candidate["runtime_tree_sha256"],
        "command": command,
    }


def run_preflight(candidate: dict[str, Any], runtime_root: Path, archive_path: Path) -> dict[str, Any]:
    out_path = OUT / "preflight" / f"{candidate['id']}.preflight.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "experiments/preflight_public_replay_intake.py",
            "--archive",
            str(archive_path),
            "--inflate-sh",
            str(runtime_root / "inflate.sh"),
            "--upstream-dir",
            str(UPSTREAM),
            "--expected-archive-sha256",
            candidate["archive_sha256"],
            "--expected-archive-size-bytes",
            str(candidate["archive_bytes"]),
            "--json-out",
            str(out_path),
            "--fail-if-not-ready",
        ],
        cwd=REPO,
        check=True,
    )
    return json.loads(out_path.read_text(encoding="utf-8"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    body = read_member(SOURCE_ARCHIVE)
    parts = split_body(body)
    dec_b, sca_b, lat_b, wrp_b = parts
    dim, delta_q, sidecar_raw = decode_current_sidecar(wrp_b)

    sidecar_decoders = {
        "split_stream": (make_split_stream, decode_split_stream),
        "bitpack_stream": (make_bitpack_stream, decode_bitpack_stream),
        "sparse_stream": (make_sparse_stream, decode_sparse_stream),
    }
    sidecar_codec_records = []
    for name, (maker, decoder) in sidecar_decoders.items():
        blob = maker(dim, delta_q)
        got_dim, got_delta = decoder(blob)
        exact = bool(np.array_equal(got_dim, dim) and np.array_equal(got_delta, delta_q))
        sidecar_codec_records.append(
            {
                "codec": name,
                "bytes": len(blob),
                "sha256": digest_bytes(blob),
                "exact_correction_roundtrip": exact,
            }
        )
        if not exact:
            raise RuntimeError(f"{name} failed exact correction roundtrip")

    rebrotli = best_current_wire(sidecar_raw)
    candidate_specs: list[dict[str, Any]] = []
    candidates_dir = OUT / "candidates"
    plans_dir = OUT / "eval_command_plans"
    candidates_dir.mkdir(exist_ok=True)
    plans_dir.mkdir(exist_ok=True)

    # Baseline normalized archive, useful to prove deterministic archive overhead.
    candidate_specs.append(
        {
            "id": "baseline_normalized_store",
            "kind": "archive_only_control",
            "parts": parts,
            "compress_type": zipfile.ZIP_STORED,
            "runtime_multi_codec": False,
        }
    )

    # Outer ZIP deflate control: logically lossless, but likely not useful on brotli payloads.
    candidate_specs.append(
        {
            "id": "baseline_outer_deflate9",
            "kind": "archive_only_outer_zip_control",
            "parts": parts,
            "compress_type": zipfile.ZIP_DEFLATED,
            "runtime_multi_codec": False,
        }
    )

    best_wrp = brotli.compress(sidecar_raw, quality=rebrotli["best"]["quality"], lgwin=rebrotli["best"]["lgwin"])
    if len(best_wrp) <= len(wrp_b):
        new_parts = [dec_b, sca_b, lat_b, best_wrp]
        candidate_specs.append(
            {
                "id": f"wrp_rebrotli_q{rebrotli['best']['quality']}_w{rebrotli['best']['lgwin']}",
                "kind": "lossless_sidecar_rebrotli",
                "parts": new_parts,
                "compress_type": zipfile.ZIP_STORED,
                "runtime_multi_codec": False,
            }
        )

    for codec_record in sidecar_codec_records:
        if codec_record["bytes"] < len(wrp_b):
            maker = sidecar_decoders[codec_record["codec"]][0]
            new_parts = [dec_b, sca_b, lat_b, maker(dim, delta_q)]
            candidate_specs.append(
                {
                    "id": codec_record["codec"],
                    "kind": "lossless_sidecar_runtime_codec",
                    "parts": new_parts,
                    "compress_type": zipfile.ZIP_STORED,
                    "runtime_multi_codec": True,
                }
            )

    candidates = []
    for spec in candidate_specs:
        candidate_body = join_body(spec["parts"])
        archive_path = candidates_dir / f"archive.{spec['id']}.zip"
        write_zip(archive_path, candidate_body, compress_type=spec["compress_type"])
        runtime_root = copy_runtime_variant(spec["id"], multi_codec=spec["runtime_multi_codec"])
        candidate = {
            "id": spec["id"],
            "kind": spec["kind"],
            "archive_path": str(archive_path.relative_to(REPO)),
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": digest_file(archive_path),
            "member_bytes": len(candidate_body),
            "member_sha256": digest_bytes(candidate_body),
            "member_delta_vs_source": len(candidate_body) - len(body),
            "archive_delta_vs_source": archive_path.stat().st_size - SOURCE_ARCHIVE.stat().st_size,
            "runtime_root": str(runtime_root.relative_to(REPO)),
            "runtime_tree_sha256": None,
            "dispatch_performed": False,
            "score_claim": False,
        }
        preflight = run_preflight(candidate, runtime_root, archive_path)
        candidate["runtime_tree_sha256"] = preflight["runtime"]["runtime_tree_sha256"]
        candidate["preflight_ready_for_exact_eval_dispatch"] = preflight["ready_for_exact_eval_dispatch"]
        candidate["preflight_path"] = str((OUT / "preflight" / f"{candidate['id']}.preflight.json").relative_to(REPO))
        plan = emit_plan(candidate, runtime_root, archive_path)
        plan_path = plans_dir / f"{candidate['id']}.exact_eval_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        candidate["exact_eval_plan_path"] = str(plan_path.relative_to(REPO))
        candidates.append(candidate)

    corrected = dim != 255
    delta_nonzero = delta_q != 0
    if not np.array_equal(corrected, delta_nonzero):
        raise RuntimeError("PR99 sidecar invariant failed: dim sentinel differs from nonzero delta")
    dim_counts = Counter(int(x) for x in dim[corrected])
    delta_counts = Counter(int(x) for x in delta_q[corrected])
    pair_indices = np.nonzero(corrected)[0]
    runs = []
    if len(pair_indices):
        start = prev = int(pair_indices[0])
        for value in pair_indices[1:]:
            value = int(value)
            if value == prev + 1:
                prev = value
            else:
                runs.append([start, prev, prev - start + 1])
                start = prev = value
        runs.append([start, prev, prev - start + 1])

    analysis = {
        "schema": "pr99_sidecar_deconstruction_v1",
        "dispatch_performed": False,
        "score_claim": False,
        "source": {
            "archive": str(SOURCE_ARCHIVE.relative_to(REPO)),
            "archive_bytes": SOURCE_ARCHIVE.stat().st_size,
            "archive_sha256": digest_file(SOURCE_ARCHIVE),
            "member_bytes": len(body),
            "member_sha256": digest_bytes(body),
            "runtime": str(SOURCE_RUNTIME.relative_to(REPO)),
            "sanitized_runtime": str(SANITIZED_RUNTIME.relative_to(REPO)),
        },
        "archive_split": {
            "decoder_brotli_bytes": len(dec_b),
            "scales_fp16_bytes": len(sca_b),
            "latents_brotli_bytes": len(lat_b),
            "sidecar_bytes": len(wrp_b),
            "length_prefix_bytes": 16,
            "member_bytes": len(body),
            "outer_zip_overhead_bytes": SOURCE_ARCHIVE.stat().st_size - len(body),
        },
        "sidecar_distribution": {
            "raw_bytes": len(sidecar_raw),
            "compressed_bytes": len(wrp_b),
            "n_pairs": int(len(dim)),
            "corrected_pairs": int(corrected.sum()),
            "uncorrected_pairs": int((~corrected).sum()),
            "corrected_fraction": float(corrected.mean()),
            "delta_abs_sum": int(np.abs(delta_q.astype(np.int16)).sum()),
            "delta_abs_mean_corrected": float(np.abs(delta_q[corrected].astype(np.int16)).mean()),
            "delta_abs_max": int(np.abs(delta_q.astype(np.int16)).max()),
            "delta_q_min": int(delta_q.min()),
            "delta_q_max": int(delta_q.max()),
            "dim_counts": {str(k): v for k, v in sorted(dim_counts.items())},
            "top_delta_counts": {str(k): v for k, v in delta_counts.most_common(16)},
            "corrected_pair_runs": runs,
            "largest_corrected_pair_runs": sorted(runs, key=lambda r: r[2], reverse=True)[:16],
        },
        "sidecar_rebrotli_search": rebrotli,
        "lossless_codec_screens": sorted(sidecar_codec_records, key=lambda r: r["bytes"]),
        "candidates": candidates,
        "recommended_exact_eval_order": [
            c["id"]
            for c in sorted(
                [c for c in candidates if c["archive_delta_vs_source"] < 0 and c["preflight_ready_for_exact_eval_dispatch"]],
                key=lambda c: c["archive_bytes"],
            )
        ],
    }
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "candidates": candidates, "recommended": analysis["recommended_exact_eval_order"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
