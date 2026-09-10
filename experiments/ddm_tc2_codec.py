"""One TC2 n600 twin tail encode and causal parse-back on pinned base-row traces.

This is a conditional tail proof: base HPAC/TC1 probabilities are the verified
move37 trace, not recomputed by a standalone public receiver. Geometry and its
online probability counts ARE regenerated from decoded prefixes. No scorer runs.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from experiments import ddm_tc2_lane_context as ref

io, tc1 = ref.io, ref.tc1
ROOT, H, W, K = ref.ROOT, ref.H, ref.W, ref.K
Y, X, GROUP = ref.Y, ref.X, ref.GROUP
SLOT = X // 64
ROW_SLOT = Y * 8 + SLOT
VALUES = (np.ones((H, W)), Y, Y * Y, X, X * Y, X * X)
POSITIONS = [np.flatnonzero(GROUP.reshape(-1) == g) for g in range(ref.G)]
ORDER = np.concatenate(POSITIONS)
MAGIC = b"TC2M\x01"


class LaneGeometry:
    """Online sufficient statistics for exactly the reviewed slotwise row fit."""

    def __init__(self, previous):
        self.plane = np.full((H, W), K, dtype=np.uint8)
        self.current = np.zeros((6, 1, H, 8))
        self.lo, self.hi = np.full((H, 8), W), np.full((H, 8), -1)
        prior = np.zeros((H, W), dtype=bool) if previous is None else previous == 1
        self.old = np.stack(
            [np.bincount(ROW_SLOT[prior], weights=v[prior], minlength=H * 8).reshape(1, H, 8) for v in VALUES]
        )
        self.old_lo, self.old_hi = np.full((H, 8), W), np.full((H, 8), -1)
        np.minimum.at(self.old_lo.reshape(-1), ROW_SLOT[prior], X[prior])
        np.maximum.at(self.old_hi.reshape(-1), ROW_SLOT[prior], X[prior])
        runs = prior.reshape(H, 8, 64)
        starts = runs & ~np.concatenate([np.zeros((H, 8, 1), dtype=bool), runs[:, :, :-1]], axis=2)
        self.ambiguous_prior = starts.sum(axis=2) > 1

    def contexts(self, positions):
        count, sy, syy, sx, sxy, sxx = [ref.above_window(v)[0] for v in self.current + self.old * 0.25]
        den = count * syy - sy * sy
        slope = np.divide(count * sxy - sy * sx, den, out=np.zeros_like(den), where=den > 0)
        my = np.divide(sy, count, out=np.zeros_like(sy), where=count > 0)
        mx = np.divide(sx, count, out=np.zeros_like(sx), where=count > 0)
        centers = mx + slope * (np.arange(H)[:, None] - my)
        variance = np.divide(sxx, count, out=np.zeros_like(sxx), where=count > 0) - mx**2
        residual = variance - slope * np.divide(sxy - my * sx, count, out=np.zeros_like(sxy), where=count > 0)
        width = np.maximum(0.5, np.sqrt(np.maximum(0, 3 * residual)))
        valid = (count >= 4) & (den > 0) & (residual <= 36) & (np.abs(slope) <= 2)
        known = self.plane.reshape(H, 8, 64)
        xx = X.reshape(H, 8, 64)
        gap = ((known < K) & (known != 1) & (xx > self.lo[:, :, None]) & (xx < self.hi[:, :, None])).any(axis=2)
        wide = (np.maximum(self.hi, self.old_hi) - np.minimum(self.lo, self.old_lo)) > 24
        valid &= ref.above_window((wide | gap | self.ambiguous_prior)[None].astype(np.int64))[0] == 0
        yy, xx = positions // W, positions % W
        distance = np.full(len(positions), np.inf)
        for slot in range(8):
            d = np.minimum(
                np.abs(xx - (centers[yy, slot] - width[yy, slot])), np.abs(xx - (centers[yy, slot] + width[yy, slot]))
            )
            distance = np.minimum(distance, np.where(valid[yy, slot], np.rint(d), np.inf))
        bins = np.searchsorted([0, 1, 2, 4, 8, 16], distance, side="left").astype(np.uint8)
        bins[~np.isfinite(distance)] = 7
        bins[(yy < 128) | (yy >= 320)] = 8
        return bins

    def observe(self, positions, symbols):
        self.plane.reshape(-1)[positions] = symbols
        lane = positions[symbols == 1]
        code = ROW_SLOT.reshape(-1)[lane]
        for j, value in enumerate(VALUES):
            np.add.at(self.current[j].reshape(-1), code, value.reshape(-1)[lane])
        np.minimum.at(self.lo.reshape(-1), code, X.reshape(-1)[lane])
        np.maximum.at(self.hi.reshape(-1), code, X.reshape(-1)[lane])


def coding(rows, phi, weights):
    freq = tc1.frequencies(rows)
    result = tc1.mix_probabilities(freq, phi[:, :, None], weights[:, None], rows)
    # A zero geometric correction must forward the original float32 row.
    result[np.all(phi == 0, axis=1)] = rows[np.all(phi == 0, axis=1)]
    return result


def setup(stage):
    ref.pin("codec_" + stage)
    gate = json.loads((ROOT / "RESULT.json").read_text())
    if not gate["encode_gate"] or not gate["full_n600"]:
        raise ValueError("full-field oracle reference has not cleared")
    work = ROOT / "codec" / stage
    work.mkdir(parents=True, exist_ok=True)
    source = io.file_fact(Path(__file__).resolve())
    binding = {
        "source": source,
        "native_wrapper": io.file_fact(io.ROUTE_B),
        "measurement_inputs": io.file_fact(ROOT / "INPUTS.json"),
        "gate": io.file_fact(ROOT / "RESULT.json"),
        "weights": io.file_fact(ROOT / "fit/causal/weights_i8.bin"),
    }
    if stage == "decode":
        binding["candidate_rider"] = io.file_fact(ROOT / "codec/encode/twin0.rider")
    binding_path = work / "INPUTS.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("codec resume binding changed")
    ref.save_json(binding_path, binding)
    route = io.load_route_b()
    build_path = work / "BUILD.json"
    if build_path.exists():
        build = json.loads(build_path.read_text())
        for key in ("base_source", "generated", "library"):
            if io.file_fact(Path(build[key]["path"])) != build[key]:
                raise ValueError("native build changed")
        library = Path(build["library"]["path"])
    else:
        library, build = io.compile_rc64(work, route, "tc2_" + stage)
        ref.save_json(build_path, build)
    weights = np.frombuffer((ROOT / "fit/causal/weights_i8.bin").read_bytes(), dtype=np.int8)
    snapshots = sorted(p for p in work.glob("state_*.npz") if p.with_suffix(".json").exists())
    state = None
    if snapshots:
        p = snapshots[-1]
        if io.file_fact(p) != json.loads(p.with_suffix(".json").read_text()):
            raise ValueError("codec checkpoint changed")
        with np.load(p, allow_pickle=False) as data:
            state = {k: data[k] for k in data.files}
    return work, binding, route, library, weights, state


def encode():
    work, binding, route, library, weights, state = setup("encode")
    encoders = [
        route.NativeRc64Encoder(library, None if state is None else state[f"encoder{j}"].tobytes()) for j in range(2)
    ]
    start = 0 if state is None else int(state["frame"][0])
    bits = np.zeros(ref.N) if state is None else state["bits"]
    for frame in range(start, ref.N):
        data = ref.retained_frame(frame)
        rows, truth = ref.source_frame(frame)
        np.testing.assert_array_equal(data["truth"], truth.reshape(-1))
        probs = coding(rows, data["phi"][1], weights)
        freq = tc1.frequencies(probs)
        bits[frame] = -np.log2(freq[np.arange(H * W), data["truth"]].astype(float) / tc1.TOTAL).sum()
        for encoder in encoders:
            encoder.encode(data["truth"][ORDER].astype(np.int32), probs[ORDER])
        values = {f"encoder{j}": np.frombuffer(enc.snapshot(), dtype=np.uint8) for j, enc in enumerate(encoders)}
        values.update(frame=np.array([frame + 1]), bits=bits.copy())
        ref.save_arrays(work / f"state_{frame + 1:04d}.npz", values)
        if (frame + 1) % 25 == 0:
            print(json.dumps({"stage": "encode", "frame": frame + 1}), flush=True)
    payloads = []
    raw_streams = []
    for j, encoder in enumerate(encoders):
        payloads.append(ref.preserve(work / f"twin{j}.envelope", encoder.finish()))
        size = int(encoder.library.rc64_encoder_size(encoder.context))
        raw = ctypes.string_at(encoder.library.rc64_encoder_data(encoder.context), size)
        raw_streams.append(raw)
        ref.preserve(work / f"twin{j}.rc64", raw)
        rider = MAGIC + (ROOT / "retained/tc1_weights.bin").read_bytes() + weights.tobytes() + raw
        ref.preserve(work / f"twin{j}.rider", rider)
    if raw_streams[0] != raw_streams[1]:
        raise ValueError("independent encoder twins differ")
    old_bytes = (ROOT / "retained/shipped.rc64").stat().st_size
    return ref.save_json(
        work / "RESULT.json",
        {
            "binding": binding,
            "axis": "[exact tail bytes, macOS-CPU scorer-free]",
            "score_claim": False,
            "frames": ref.N,
            "positions": ref.N * H * W,
            "twins_byte_identical": True,
            "payloads": payloads,
            "source_raw_stream_bytes": old_bytes,
            "candidate_raw_stream_bytes": len(raw_streams[0]),
            "source_rider_bytes": old_bytes + 40,
            "candidate_rider_bytes": len(raw_streams[0]) + 45,
            "net_tail_saved_bytes": old_bytes - len(raw_streams[0]) - 5,
            "exact_frequency_codelength_bits": float(bits.sum()),
            "candidate_rider": io.file_fact(work / "twin0.rider"),
            "archive_created": False,
            "parseback": "required next",
            "source_field_sha256": ref.FIELD_SHA,
        },
    )


def decode(stop):
    work, binding, route, library, weights, state = setup("decode")
    rider = (ROOT / "codec/encode/twin0.rider").read_bytes()
    if (
        rider[:5] != MAGIC
        or rider[5:40] != (ROOT / "retained/tc1_weights.bin").read_bytes()
        or rider[40:45] != weights.tobytes()
    ):
        raise ValueError("counted rider parse failed")
    envelope = b"R6D1" + rider[45:]
    ref.preserve(work / "parsed.envelope", envelope)
    decoder = route.NativeRc64Decoder(library, envelope if state is None else state["decoder"].tobytes())
    start = 0 if state is None else int(state["frame"][0])
    if not start <= stop <= ref.N:
        raise ValueError("invalid decoder resume boundary")
    count = np.zeros((K * ref.BINS, K), dtype=np.int64) if state is None else state["count"]
    expected = np.zeros_like(count) if state is None else state["expected"]
    previous = None if state is None else state["previous"]
    for frame in range(start, stop):
        rows, source = ref.source_frame(frame)
        prepared = ref.retained_frame(frame)
        geometry = LaneGeometry(previous)
        table = tc1.log2_fixed(np.clip((count + 0.5) / (expected.astype(float) / tc1.TOTAL + 0.5), 1 / 16, 16))
        ctx = np.empty(H * W, dtype=np.uint8)
        for positions in POSITIONS:
            bins = geometry.contexts(positions)
            np.testing.assert_array_equal(bins, prepared["maps"][1].reshape(-1)[positions])
            freq = tc1.frequencies(rows[positions])
            code = freq.argmax(axis=1) * ref.BINS + bins
            phi = table[code].copy()
            phi[bins == 8] = 0
            np.testing.assert_array_equal(phi, prepared["phi"][1, positions])
            symbols = decoder.decode(None, coding(rows[positions], phi, weights)).astype(np.uint8)
            geometry.observe(positions, symbols)
            ctx[positions] = bins
        plane = geometry.plane
        np.testing.assert_array_equal(plane, source)
        freq = tc1.frequencies(rows)
        code = freq.argmax(axis=1) * ref.BINS + ctx
        count += np.bincount(code * K + plane.reshape(-1), minlength=K * ref.BINS * K).reshape(-1, K)
        for k in range(K):
            expected[:, k] += np.bincount(code, weights=freq[:, k], minlength=K * ref.BINS).astype(np.int64)
        ref.preserve(work / "frames" / f"frame_{frame:04d}.u8", plane.tobytes())
        ref.save_arrays(
            work / f"state_{frame + 1:04d}.npz",
            {
                "frame": np.array([frame + 1]),
                "decoder": decoder.get_compressed(),
                "count": count,
                "expected": expected,
                "previous": plane,
            },
        )
        previous = plane
        if (frame + 1) % 10 == 0 or frame + 1 == stop:
            print(json.dumps({"stage": "decode", "frame": frame + 1}), flush=True)
    digest = hashlib.sha256()
    for frame in range(stop):
        digest.update((work / "frames" / f"frame_{frame:04d}.u8").read_bytes())
    if stop == ref.N and (digest.hexdigest() != ref.FIELD_SHA or not decoder.is_empty()):
        raise ValueError("full field digest or symbol count differs")
    return ref.save_json(
        work / ("RESULT.json" if stop == ref.N else f"PARTIAL_{stop:04d}.json"),
        {
            "binding": binding,
            "axis": "[conditional tail parse-back, macOS-CPU scorer-free]",
            "score_claim": False,
            "frames": stop,
            "positions": stop * H * W,
            "decoded_field_sha256": digest.hexdigest(),
            "full_n600_identity": stop == ref.N,
            "all_group_geometry_and_phi_parity": True,
            "base_probability_source": "verified shipped trace, justified by same-prefix induction; HPAC not rerun here",
            "standalone_public_receiver_proof": False,
            "full_frame_inflate_output_parity_missing": True,
            "cross_host_bin_parity_measured": False,
            "resumed_from_frame": start,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("encode", "decode"))
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=ref.N)
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT:
        parser.error("assigned TC2 root required")
    result = encode() if args.stage == "encode" else decode(args.stop_after)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
