"""ddm_gdc4 stage-00b: price the exact run-native packet with a real coder.

Encodes the move-44 token field into the arm's run-native streams, races the
three physical coders on every stream independently, proves exact decode back
to the source bytes, and reports the counted packet size against the
``94,010 B`` generator door.

This is a zero-training, zero-scorer, byte-only measurement. It establishes the
DESCRIPTION INTERCEPT of the run-native representation: the number a learned
endpoint generator would have to beat with its own counted state.

``research_only=true`` · ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac.gdc4_run_native_endpoint import (
    STREAM_NAMES,
    V2_STREAM_NAMES,
    decode_field_runs,
    decode_field_transitions,
    encode_field_runs,
    encode_field_transitions,
)

CODECS = {
    "v1_run_mode": (encode_field_runs, decode_field_runs, STREAM_NAMES),
    "v2_transition_edit": (
        encode_field_transitions,
        decode_field_transitions,
        V2_STREAM_NAMES,
    ),
}

FIELD_SHAPE = (600, 384, 512)
DOOR_TOTAL_B = 94_010
SHIPPED_TAIL_B = 119_969


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def race_coders(payload: bytes) -> dict[str, object]:
    """Race the three physical coders; return the winner and every candidate."""
    import lzma

    candidates: dict[str, bytes] = {}
    try:
        import brotli

        candidates["brotli_q11"] = brotli.compress(payload, quality=11)
    except Exception as exc:  # pragma: no cover - environment dependent
        candidates_error = str(exc)
    else:
        candidates_error = ""
    lzma_filters = [
        {
            "id": lzma.FILTER_LZMA2,
            "preset": 9 | lzma.PRESET_EXTREME,
        }
    ]
    candidates["lzma2_extreme"] = lzma.compress(
        payload, format=lzma.FORMAT_RAW, filters=lzma_filters
    )
    candidates["zlib9"] = zlib.compress(payload, 9)

    sizes = {name: len(blob) for name, blob in candidates.items()}
    winner = min(sizes, key=lambda name: sizes[name])
    return {
        "raw_bytes": len(payload),
        "raw_sha256": sha256_bytes(payload),
        "sizes": sizes,
        "winner": winner,
        "winner_bytes": sizes[winner],
        "winner_sha256": sha256_bytes(candidates[winner]),
        "coder_error": candidates_error,
        "_payloads": candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames", type=int, default=FIELD_SHAPE[0])
    parser.add_argument("--prefer-previous-frame", action="store_true")
    parser.add_argument("--codec", default="v1_run_mode", choices=sorted(CODECS))
    parser.add_argument("--retain-streams", action="store_true")
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args(argv)

    field_path = Path(args.field)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_frames = int(args.frames)
    shape = (n_frames, FIELD_SHAPE[1], FIELD_SHAPE[2])
    field = np.memmap(field_path, dtype=np.uint8, mode="r", shape=FIELD_SHAPE)
    view = np.asarray(field[:n_frames])  # SUBSET_SELECTION_OK: the codec is CAUSAL over frames (each frame's reference is the previous frame), so a contiguous prefix is the only coherent subset; --frames exists solely for local timing/round-trip smokes and every reported number in this arm is the full n600.

    encoder, decoder, stream_names = CODECS[args.codec]
    kwargs = (
        {"prefer_previous_frame": args.prefer_previous_frame}
        if args.codec == "v1_run_mode"
        else {}
    )
    started = time.time()
    streams = encoder(view, **kwargs)
    encode_s = time.time() - started

    t0 = time.time()
    rebuilt = decoder(streams, shape)
    decode_s = time.time() - t0
    exact = bool(np.array_equal(rebuilt, view))
    if not exact:
        raise SystemExit("FAIL: run-native decode is not byte-identical to the field")

    # Determinism repeat: re-encode must be byte-identical.
    repeat = encoder(view, **kwargs)
    deterministic = all(repeat[name] == streams[name] for name in stream_names)

    hist_key = "mode" if args.codec == "v1_run_mode" else "op"
    mode_hist = np.bincount(
        np.frombuffer(streams[hist_key], dtype=np.uint8), minlength=5
    ).tolist()

    per_stream: dict[str, dict[str, object]] = {}
    packet_bytes = 0
    for name in stream_names:
        race = race_coders(streams[name])
        payloads = race.pop("_payloads")
        if args.retain_streams:
            (out_dir / f"stream_{name}.raw").write_bytes(streams[name])
            for coder, blob in payloads.items():
                (out_dir / f"stream_{name}.{coder}").write_bytes(blob)
        per_stream[name] = race
        packet_bytes += int(race["winner_bytes"])

    # Header cost: five uvarint stream lengths, counted honestly.
    header_bytes = 0
    for name in stream_names:
        length = int(per_stream[name]["winner_bytes"])
        header_bytes += max(1, (length.bit_length() + 6) // 7)
    packet_with_header = packet_bytes + header_bytes

    result = {
        "arm": "ddm_gdc4",
        "stage": "stage00b_run_native_packet",
        "research_only": True,
        "score_claim": False,
        "promotable": False,
        "axis": "[macOS-CPU byte-only n600]",
        "seed": int(args.seed),
        "frames": n_frames,
        "field": {
            "path": str(field_path),
            "sha256": sha256_file(field_path) if n_frames == FIELD_SHAPE[0] else None,
            "shape": list(shape),
        },
        "prefer_previous_frame": bool(args.prefer_previous_frame),
        "exact_decode": exact,
        "deterministic_reencode": deterministic,
        "encode_s": encode_s,
        "decode_s": decode_s,
        "codec": args.codec,
        "symbol_histogram": mode_hist,
        "streams": per_stream,
        "counted_packet_bytes": packet_with_header,
        "counted_packet_streams_bytes": packet_bytes,
        "counted_packet_header_bytes": header_bytes,
        "door_total_bytes": DOOR_TOTAL_B,
        "shipped_tail_bytes": SHIPPED_TAIL_B,
        "vs_door_bytes": packet_with_header - DOOR_TOTAL_B,
        "vs_door_ratio": packet_with_header / DOOR_TOTAL_B,
        "vs_shipped_tail_ratio": packet_with_header / SHIPPED_TAIL_B,
        "note": (
            "This packet is EXACT: it needs no residual. Any learned endpoint "
            "generator must beat this counted intercept with its own state."
        ),
    }

    out_path = out_dir / "RESULT.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    os.replace(tmp, out_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
