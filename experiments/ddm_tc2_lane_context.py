#!/usr/bin/env python3
"""Full-n600 fixed-base lane-context measurement; never a contest score.

The GT map is a noncausal reference, NOT a universal upper bound. The causal
map fits separate 64-column local lane tracks, rejecting ambiguous tracks.
All measured arrays, fitted weights and coder payloads are retained on SSD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from scipy.ndimage import distance_transform_cdt
from scipy.special import logsumexp

from experiments import ddm_jg2_tail_reencode as io
from experiments import ddm_tc1_mixer_codec as tc1

ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_tc2_lane_context_map")
SOURCE = Path("/Volumes/VertigoDataTier/pact/ddm_bnd2_segment_code/generation2")
LIVE = Path("/Volumes/VertigoDataTier/pact/ddm_cmp2_compose/candidate_runtime")
POINTER = REPO / ".omx/state/canonical_frontier_pointer.json"
ARCHIVE_SHA = "670d38d05eb142fec9579337e21d7c6522592769ec00c0271aa971ee018ce6bc"
FIELD_SHA = "361cc6c9749fdec1381936836c9b45f4e04702f02eed9f8ea5343b1afa957b94"
N, H, W, K = 600, 384, 512, 5
BINS = 9  # distance 0,1,2,3-4,5-8,9-16,>16; unavailable; outside band
AXIS = "[macOS-CPU advisory / real full-n600 counts, scorer-free]"
Y, X = np.indices((H, W))
GROUP = (X % 64 + 2 * (Y % 64)).astype(np.int64)
G = 190
WINDOW = 16
SEED = 20260910
KEEP = 2.0**-14


def storage(need=0):
    if shutil.disk_usage(ROOT).free < 40 * 1024**3 + need:
        raise RuntimeError("STORAGE_RESERVE: preserve bytes and stop")


def save_json(path, value):
    storage()
    io.atomic_json(path, value)
    return value


def save_arrays(path, values):
    storage(sum(v.nbytes for v in values.values()))
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(values) or any(not np.array_equal(old[k], v) for k, v in values.items()):
                raise ValueError(f"immutable array mismatch: {path}")
    else:
        io.atomic_npz(path, values)
    save_json(path.with_suffix(".json"), io.file_fact(path))


def preserve(path, value):
    storage(len(value))
    io.persist_immutable_bytes(path, value, label="TC2 retained payload")
    return io.file_fact(path)


def pin(stage):
    """Read the live pointer at every stage and refuse changed input custody."""
    ROOT.mkdir(parents=True, exist_ok=True)
    storage()
    pointer_bytes = POINTER.read_bytes()
    pointer = json.loads(pointer_bytes)
    # Charter explicitly permits the named move-37 field or a successor.
    # Reread/snapshot live state, but never transfer this fixed-field result.
    pointer_sha = hashlib.sha256(pointer_bytes).hexdigest()
    preserve(ROOT / "retained" / f"pointer_{stage}_{pointer_sha}.json", pointer_bytes)
    print(
        json.dumps({"live_pointer": pointer["effective_frontier"], "measurement_archive_sha": ARCHIVE_SHA}), flush=True
    )
    archive = io.file_fact(LIVE / "archive.zip")
    if archive["sha256"] != ARCHIVE_SHA:
        raise ValueError("archive differs from pointer")
    trace = json.loads((SOURCE / "trace/RESULT.json").read_text())
    if (
        trace["frames"] != N
        or not trace["full_control_byte_identical"]
        or not trace["twin_byte_identical"]
        or trace["field_sha256"] != FIELD_SHA
    ):
        raise ValueError("source trace lacks full current-field control")
    field = io.file_fact(SOURCE / "field.u8")
    if field["sha256"] != FIELD_SHA:
        raise ValueError("source field differs")
    if not (ROOT / "runtime").exists():
        shutil.copytree(
            LIVE, ROOT / "runtime", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.dylib", "*.so")
        )
    residual, _, _ = io.load_runtime(ROOT / "runtime")
    parts = residual.read_residual_archive(ROOT / "runtime/archive.zip")
    for name, value in vars(parts).items():
        if isinstance(value, (bytes, bytearray)):
            preserve(ROOT / "retained" / f"reader_{name}.bin", bytes(value))
    raw = parts.token_stream
    preserve(ROOT / "retained/shipped.rc64", raw)
    preserve(ROOT / "retained/tc1_weights.bin", bytes(parts.tc1_weights))
    envelope = b"R6D1" + raw
    envelope += b"\0" * (-len(envelope) % 4)
    preserve(ROOT / "retained/shipped.envelope", envelope)
    if any(Path(p["path"]).read_bytes() != envelope for p in trace["payloads"]):
        raise ValueError("current public reader stream differs from retained trace twins")
    binding = {
        "archive": archive,
        "field": field,
        "trace": io.file_fact(SOURCE / "trace/RESULT.json"),
        "seed": SEED,
        "score_claim": False,
        "axis": AXIS,
        "source": io.file_fact(Path(__file__)),
        "upstream": io.file_fact(REPO / "upstream/evaluate.py"),
        "dependencies": [io.file_fact(Path(module.__file__)) for module in (io, tc1)],
    }
    path = ROOT / "INPUTS.json"
    if path.exists():
        if json.loads(path.read_text())["binding"] != binding:
            raise ValueError("restart source/input binding changed; retain and rebind explicitly")
    else:
        save_json(
            path,
            {
                "binding": binding,
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "retention": "all payloads retained; no deletion or movement; scratch only atomic writes",
                "free_bytes": shutil.disk_usage(ROOT).free,
            },
        )
    return binding


def distance_bins(distance, valid=None):
    result = np.searchsorted(np.array([0, 1, 2, 4, 8, 16]), distance, side="left").astype(np.uint8)
    if valid is not None:
        result[~valid] = 7
    result[(Y < 128) | (Y >= 320)] = 8
    return result


def oracle_map(plane):
    """Unsigned Manhattan distance to GT Lane/Road interfaces, both endpoints."""
    edge = np.zeros((H, W), dtype=bool)
    for axis in (0, 1):
        a, b = (plane[:-1], plane[1:]) if axis == 0 else (plane[:, :-1], plane[:, 1:])
        cross = ((a == 1) & (b == 0)) | ((a == 0) & (b == 1))
        if axis == 0:
            edge[:-1] |= cross
            edge[1:] |= cross
        else:
            edge[:, :-1] |= cross
            edge[:, 1:] |= cross
    if not edge.any():
        return distance_bins(np.zeros((H, W)), np.zeros((H, W), dtype=bool))
    return distance_bins(distance_transform_cdt(~edge, metric="taxicab"))


def above_window(value):
    """Sum exactly rows max(0,y-16)..y-1; current row never enters."""
    prefix = np.concatenate([np.zeros_like(value[:, :1]), np.cumsum(value, axis=1)], axis=1)
    return prefix[:, np.arange(H)] - prefix[:, np.maximum(0, np.arange(H) - WINDOW)]


def causal_map(plane, previous):
    """Local degree-one lane fits from legal above-row pixels and prior plane.

    Eight separate horizontal slots prevent pooling the road's two lane lines.
    A fit spanning more than 24 columns or with residual RMS > 6 is unavailable.
    No learned coefficient or per-frame table is transmitted. Fits freeze before
    each HPAC group; unknown current-group and later-group pixels never enter.
    """
    slot = X // 64
    code = (GROUP * H + Y) * 8 + slot
    mask = plane == 1
    prior = np.zeros_like(mask) if previous is None else previous == 1
    moments = []
    for value in (np.ones((H, W)), Y, Y * Y, X, X * Y, X * X):
        current = np.bincount(code[mask], weights=value[mask], minlength=G * H * 8).reshape(G, H, 8)
        # Strict group prefix; the full-plane argument is an encoder convenience.
        current = np.concatenate([np.zeros_like(current[:1]), np.cumsum(current, axis=0)[:-1]], axis=0)
        old = np.bincount((Y * 8 + slot)[prior], weights=value[prior], minlength=H * 8).reshape(1, H, 8)
        moments.append(above_window(current + old * 0.25))
    count, sy, syy, sx, sxy, sxx = moments
    den = count * syy - sy * sy
    slope = np.divide(count * sxy - sy * sx, den, out=np.zeros_like(den), where=den > 0)
    mean_y = np.divide(sy, count, out=np.zeros_like(sy), where=count > 0)
    mean_x = np.divide(sx, count, out=np.zeros_like(sx), where=count > 0)
    centers = mean_x + slope * (np.arange(H)[None, :, None] - mean_y)
    variance = np.divide(sxx, count, out=np.zeros_like(sxx), where=count > 0) - mean_x**2
    residual = variance - slope * np.divide(sxy - mean_y * sx, count, out=np.zeros_like(sxy), where=count > 0)
    width = np.maximum(0.5, np.sqrt(np.maximum(0, 3 * residual)))
    valid = (count >= 4) & (den > 0) & (residual <= 36) & (np.abs(slope) <= 2)
    # Exclude a slot if its observed/prior row contains separated Lane runs.
    # The min/max test is computed on the identical strict causal information set.
    lo = np.full((G, H, 8), W, dtype=np.int64)
    hi = np.full((G, H, 8), -1, dtype=np.int64)
    np.minimum.at(lo.reshape(-1), code[mask], X[mask])
    np.maximum.at(hi.reshape(-1), code[mask], X[mask])
    lo = np.concatenate([np.full_like(lo[:1], W), np.minimum.accumulate(lo, axis=0)[:-1]])
    hi = np.concatenate([np.full_like(hi[:1], -1), np.maximum.accumulate(hi, axis=0)[:-1]])
    ambiguous = np.zeros_like(lo, dtype=bool)
    for dx in range(64):
        # A known non-Lane symbol between two known Lane pixels separates runs.
        pixel_x = X[:, dx::64][None]
        known = np.arange(G)[:, None, None] > GROUP[:, dx::64][None]
        ambiguous |= known & (plane[:, dx::64][None] != 1) & (pixel_x > lo) & (pixel_x < hi)
    prior_runs = prior.reshape(H, 8, 64)
    starts = prior_runs & ~np.concatenate([np.zeros((H, 8, 1), dtype=bool), prior_runs[:, :, :-1]], axis=2)
    ambiguous |= (starts.sum(axis=2) > 1)[None]
    old_lo, old_hi = np.full((H, 8), W), np.full((H, 8), -1)
    np.minimum.at(old_lo.reshape(-1), (Y * 8 + slot)[prior], X[prior])
    np.maximum.at(old_hi.reshape(-1), (Y * 8 + slot)[prior], X[prior])
    wide = (np.maximum(hi, old_hi) - np.minimum(lo, old_lo)) > 24
    valid &= above_window((wide | ambiguous).astype(np.int64)) == 0
    distance = np.full((H, W), np.inf)
    for track in range(8):
        center = centers[GROUP, Y, track]
        half = width[GROUP, Y, track]
        ok = valid[GROUP, Y, track]
        d = np.minimum(np.abs(X - (center - half)), np.abs(X - (center + half)))
        distance = np.minimum(distance, np.where(ok, np.rint(d), np.inf))
    return distance_bins(distance, np.isfinite(distance))


def verify_causality(plane, previous):
    """Mutation of every unavailable current/later group cannot alter a context."""
    original = causal_map(plane, previous)
    checks = []
    for group in (0, 31, 63, 94, 126, 158, 189):
        changed = plane.copy()
        changed[group <= GROUP] = (changed[group <= GROUP] + 1) % K
        actual = causal_map(changed, previous)
        np.testing.assert_array_equal(original[group == GROUP], actual[group == GROUP])
        checks.append(group)
    return {"groups": checks, "unavailable_group_mutation_invariant": True}


def source_frame(frame):
    path = SOURCE / "trace/frames" / f"frame_{frame:04d}.npz"
    receipt = json.loads(path.with_suffix(".json").read_text())
    if io.file_fact(path) != receipt["payload"] or receipt["source_field_sha256"] != FIELD_SHA:
        raise ValueError("row custody mismatch")
    with np.load(path, allow_pickle=False) as data:
        return data["rows"], data["tokens"]


def prepare(stop):
    binding = pin("prepare")
    from experiments import ddm_jg1_seg_solve as seg
    from experiments import ddm_up2_shipping_pose_solve as up

    up.verify_gt_lineage(axis="contest_cuda", declared_lineage=up.LINEAGE_DALI)
    gt = seg.load_gt_seg_labels(up.LINEAGE_DALI)
    gt_fact = preserve(ROOT / "retained/gt_dali_seg.u8", gt.tobytes())
    starts = sorted(p for p in (ROOT / "prepare").glob("state_*.npz") if p.with_suffix(".json").exists())
    counts = np.zeros((2, K * BINS, K), dtype=np.int64)
    expected = np.zeros_like(counts)
    hist = np.zeros((3, K * 64 * BINS, K), dtype=np.int64)
    totals = np.zeros(5)
    start = 0
    if starts:
        if io.file_fact(starts[-1]) != json.loads(starts[-1].with_suffix(".json").read_text()):
            raise ValueError("prepare checkpoint digest differs")
        with np.load(starts[-1], allow_pickle=False) as data:
            counts, expected, hist, totals = (data[k] for k in ("counts", "expected", "hist", "totals"))
            start = int(data["frame"][0])
    if start > stop:
        raise ValueError("requested stop precedes saved checkpoint")
    field = np.memmap(SOURCE / "field.u8", dtype=np.uint8, mode="r", shape=(N, H, W))
    for frame in range(start, stop):
        rows, plane = source_frame(frame)
        np.testing.assert_array_equal(plane, field[frame])
        previous = None if not frame else field[frame - 1]
        maps = np.stack([oracle_map(gt[frame]), causal_map(plane, previous)])
        if frame in (0, 317, 599):
            save_json(ROOT / "prepare" / f"CAUSAL_CONTROL_{frame:04d}.json", verify_causality(plane, previous))
        freq = tc1.frequencies(rows)
        arg = freq.argmax(axis=1)
        truth = plane.reshape(-1)
        phi = np.zeros((2, H * W, K), dtype=np.int16)
        bits = -np.log2(freq[np.arange(H * W), truth].astype(float) / tc1.TOTAL)
        keep = (1 - freq.max(axis=1).astype(float) / tc1.TOTAL >= KEEP) | (truth != arg)
        bucket = arg * 64 + np.minimum((-2 * np.log2(1 - freq.max(axis=1).astype(float) / tc1.TOTAL)).astype(int), 63)
        hist[0] += np.bincount((bucket * BINS) * K + truth, minlength=K * 64 * BINS * K).reshape(-1, K)
        for j in range(2):
            code = arg * BINS + maps[j].reshape(-1)
            ratio = (counts[j] + 0.5) / (expected[j].astype(float) / tc1.TOTAL + 0.5)
            phi[j] = tc1.log2_fixed(np.clip(ratio, 1 / 16, 16))[code]
            phi[j, maps[j].reshape(-1) == 8] = 0
            counts[j] += np.bincount(code * K + truth, minlength=K * BINS * K).reshape(-1, K)
            for k in range(K):
                expected[j, :, k] += np.bincount(code, weights=freq[:, k], minlength=K * BINS).astype(np.int64)
            hcode = bucket * BINS + maps[j].reshape(-1)
            hist[j + 1] += np.bincount(hcode * K + truth, minlength=K * 64 * BINS * K).reshape(-1, K)
        save_arrays(
            ROOT / "prepare/frames" / f"frame_{frame:04d}.npz",
            {"phi": phi, "maps": maps, "keep": keep, "truth": truth, "freq": freq.astype(np.uint32)},
        )
        totals += [bits.sum(), bits[keep].sum(), bits[~keep].sum(), keep.sum(), ((truth == 0) | (truth == 1)).sum()]
        # Every completed frame is a full crash-resume boundary.
        save_arrays(
            ROOT / "prepare" / f"state_{frame + 1:04d}.npz",
            {"frame": np.array([frame + 1]), "counts": counts, "expected": expected, "hist": hist, "totals": totals},
        )
        print(json.dumps({"stage": "prepare", "frame": frame + 1, "baseline_bits": totals[0]}), flush=True)
    from experiments.ddm_tc1_context_statistics import entropy

    return save_json(
        ROOT / "prepare" / ("RESULT.json" if stop == N else f"PARTIAL_{stop:04d}.json"),
        {
            "binding": binding,
            "gt": gt_fact,
            "frames": stop,
            "positions": stop * H * W,
            "totals": totals.tolist(),
            "conditional_entropy": [entropy(x) for x in hist],
            "labels": [
                "shipped_full_probability_bucket",
                "plus_gt_lane_road_distance",
                "plus_causal_lane_fit_distance",
            ],
            "universal_upper_bound": False,
            "full_n600": stop == N,
        },
    )


def retained_frame(frame):
    path = ROOT / "prepare/frames" / f"frame_{frame:04d}.npz"
    if io.file_fact(path) != json.loads(path.with_suffix(".json").read_text()):
        raise ValueError("TC2 frame digest differs")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def fit():
    binding = pin("fit")
    prepared = json.loads((ROOT / "prepare/RESULT.json").read_text())
    if prepared["positions"] != N * H * W or not prepared["full_n600"]:
        raise ValueError("fitting requires all 600 pairs")
    chunks = {k: [] for k in ("p", "phi", "truth", "arg")}
    for frame in range(N):
        data = retained_frame(frame)
        keep = data["keep"]
        p = data["freq"][keep].astype(float) / tc1.TOTAL
        chunks["p"].append(p)
        chunks["phi"].append(data["phi"][:, keep])
        chunks["truth"].append(data["truth"][keep])
        chunks["arg"].append(p.argmax(axis=1))
    arrays = {k: np.concatenate(v, axis=1 if k == "phi" else 0) for k, v in chunks.items()}
    del chunks
    result = []
    for j, name in enumerate(("oracle", "causal")):
        folder = ROOT / "fit" / name
        folder.mkdir(parents=True, exist_ok=True)
        weights = np.zeros(K)
        previous = sorted(folder.glob("ITER_*.json"))
        start = 0
        if previous:
            old = json.loads(previous[-1].read_text())
            weights = np.array(old["weights"])
            start = old["iteration"] + 1
        logp = np.log(arrays["p"])
        x = arrays["phi"][j].astype(float) * math.log(2) / tc1.Q
        arg, truth = arrays["arg"], arrays["truth"]
        index = np.arange(len(arg))

        def objective(w, logp=logp, x=x, arg=arg, truth=truth, index=index):
            z = logp + x * w[arg, None]
            norm = logsumexp(z, axis=1)
            p = np.exp(z - norm[:, None])
            mean = (p * x).sum(axis=1)
            g = np.bincount(arg, weights=mean - x[index, truth], minlength=K)
            h = np.bincount(arg, weights=(p * (x - mean[:, None]) ** 2).sum(axis=1), minlength=K)
            return float((norm - z[index, truth]).sum()), g, h

        for iteration in range(start, start + 80):
            loss, gradient, hessian = objective(weights)
            gap = np.sum(gradient * np.where(gradient >= 0, weights + 4, weights - 127 / 32))
            state = {
                "iteration": iteration,
                "weights": weights.tolist(),
                "loss_nats": loss,
                "gap_bits": float(gap / math.log(2)),
                "gradient": gradient.tolist(),
            }
            save_json(folder / f"ITER_{iteration:04d}.json", state)
            if gap / math.log(2) < 0.05:
                break
            direction = -gradient / np.maximum(hessian, 1e-12)
            accepted = False
            for step in range(25):
                candidate = np.clip(weights + direction * (2.0**-step), -4, 127 / 32)
                candidate_loss = objective(candidate)[0]
                save_json(
                    folder / f"TRIAL_{iteration:04d}_{step:02d}.json",
                    {"weights": candidate.tolist(), "loss_nats": candidate_loss},
                )
                if candidate_loss <= loss + 1e-4 * float(np.dot(gradient, candidate - weights)):
                    weights = candidate
                    accepted = True
                    break
            if not accepted:
                raise ValueError("fit did not descend")
        else:
            raise ValueError("fit certificate not converged; resume from retained iterates")
        q = np.clip(np.rint(weights * 32), -128, 127).astype(np.int8)
        weight_fact = preserve(folder / "weights_i8.bin", q.tobytes())
        result.append(
            {
                "name": name,
                "weights": weights.tolist(),
                "quantized": q.tolist(),
                "weight_payload": weight_fact,
                "optimistic_gain_bytes": (prepared["totals"][0] - (loss - gap) / math.log(2)) / 8,
                "omitted_credit_bytes": prepared["totals"][2] / 8,
                "gap_bits": state["gap_bits"],
            }
        )
    return save_json(
        ROOT / "fit/RESULT.json",
        {
            "binding": binding,
            "rows": result,
            "fixed_base": True,
            "extra_coefficient_count": 5,
            "universal_upper_bound": False,
            "score_claim": False,
        },
    )


def totals():
    binding = pin("totals")
    fits = json.loads((ROOT / "fit/RESULT.json").read_text())["rows"]
    total = np.zeros((5, 3))  # base, oracle float/int8, causal float/int8; all/LR/band LR
    for frame in range(N):
        checkpoint = ROOT / "totals" / f"frame_{frame:04d}.npz"
        if checkpoint.with_suffix(".json").exists():
            if io.file_fact(checkpoint) != json.loads(checkpoint.with_suffix(".json").read_text()):
                raise ValueError("total checkpoint digest differs")
            with np.load(checkpoint, allow_pickle=False) as old:
                total += old["bits"]
            continue
        data = retained_frame(frame)
        truth, freq = data["truth"], data["freq"]
        arg = freq.argmax(axis=1)
        logp = np.log(freq.astype(float) / tc1.TOTAL)
        ix = np.arange(H * W)
        costs = [-logp[ix, truth] / math.log(2)]
        for j in range(2):
            for w in (np.array(fits[j]["weights"]), np.array(fits[j]["quantized"]) / 32):
                z = logp + data["phi"][j] * math.log(2) / tc1.Q * w[arg, None]
                costs.append((logsumexp(z, axis=1) - z[ix, truth]) / math.log(2))
        lr = (truth == 0) | (truth == 1)
        band = lr & (Y.reshape(-1) >= 128) & (Y.reshape(-1) < 320)
        per_frame = np.array([[v.sum(), v[lr].sum(), v[band].sum()] for v in costs])
        save_arrays(ROOT / "totals" / f"frame_{frame:04d}.npz", {"bits": per_frame})
        total += per_frame
        if (frame + 1) % 25 == 0:
            print(json.dumps({"stage": "totals", "frames": frame + 1, "bits": total[:, 0].tolist()}), flush=True)
    gain = (total[0, 0] - total[1, 0]) / 8
    gate = (
        "ENCODE"
        if gain >= 3000
        else "FIXED_EXTENSION_BELOW_BAR"
        if fits[0]["optimistic_gain_bytes"] < 3000
        else "THRESHOLD_UNCERTAIN"
    )
    return save_json(
        ROOT / "RESULT.json",
        {
            "binding": binding,
            "axis": AXIS,
            "score_claim": False,
            "full_n600": True,
            "positions": N * H * W,
            "labels": ["shipped", "oracle_float", "oracle_int8", "causal_float", "causal_int8"],
            "columns": ["all_symbols", "lane_and_road_symbols", "lane_and_road_in_rows_128_319"],
            "codelength_bits": total.tolist(),
            "reference_gain_bytes": (total[0, 0] - total[1, 0]) / 8,
            "causal_gain_bytes": (total[0, 0] - total[3, 0]) / 8,
            "encode_gate": bool((total[0, 0] - total[1, 0]) / 8 >= 3000),
            "gate_disposition": gate,
            "oracle_fixed_family_optimistic_gain_bytes": fits[0]["optimistic_gain_bytes"],
            "codelength_is_not_payload_price": True,
            "universal_upper_bound": False,
            "scope": "fixed shipped TC1 base plus one causal online ratio feature per coding winner; no refit of original 35 weights",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pin", "prepare", "fit", "totals"))
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=N)
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT or not 1 <= args.stop_after <= N:
        parser.error("assigned root and 1..600 frame boundary required")
    np.random.seed(SEED)
    result = {"pin": lambda: pin("initial"), "prepare": lambda: prepare(args.stop_after), "fit": fit, "totals": totals}[
        args.stage
    ]()
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
