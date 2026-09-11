# SPDX-License-Identifier: MIT
"""Amplitude-response of a post-render boundary treatment on the move-44 field.

The producer's operators replace every token-edge pixel outright.  MEASURED in
the scorer's own 384x512 plane that is a mean |dRGB| of 27 of 255 levels at
guided and 4 at sdf, applied to 2,287,200 edge pixels of which only 0.53% are
actually wrong.  This runner traces the benefit/harm curve as a function of the
amplitude bound tau, by clamping the retained treated frame's delta against the
retained baseline frame.  tau = 255 reproduces the producer exactly; tau < 0 is
the sign control that moves AWAY from the operator, which separates a directional
signal from plain out-of-distribution sensitivity.

No new render is needed: every candidate is integer arithmetic on two retained
uint8 payloads.  A tau that is a power of two is generic; a tau chosen by
measuring this video is fitted and would have to ship as a counted archive byte
(25/37,545,489 = 6.66e-7 S per byte), never as a free receiver literal.

research_only=true, score_claim=false, axis [macOS-CPU advisory].
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "upstream")]

N, H, W, CH, CW = 600, 384, 512, 874, 1164
CHUNK = 5
ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_rbf1/retained")
GT = Path("/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt")
POINTER_BYTES = 180406
POINTER_D_POSE = 4.59e-06


def clamp_delta(base: np.ndarray, treated: np.ndarray, tau: int) -> np.ndarray:
    """Bound the operator's per-channel move; negative tau reverses its sign."""
    delta = treated.astype(np.int16) - base.astype(np.int16)
    bound = abs(tau)
    limited = np.clip(delta, -bound, bound)
    if tau < 0:
        limited = -limited
    return np.clip(base.astype(np.int16) + limited, 0, 255).astype(np.uint8)


def load_scorer():
    import modules
    net = modules.DistortionNet().eval()
    net.load_state_dicts(modules.posenet_sd_path, modules.segnet_sd_path, torch.device("cpu"))
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net, modules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="guided")
    parser.add_argument("--taus", default="1,2,4,8,16,255,-8")
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default=str(ROOT / "amplitude"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    taus = [int(t) for t in args.taus.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gt = torch.load(GT, map_location="cpu", weights_only=False)
    labels = gt["seg"].numpy().astype(np.uint8)
    poses = gt["pose"].numpy().astype(np.float64)
    net, modules = load_scorer()
    rng = np.random.default_rng(args.seed)
    pairs = sorted(int(p) for p in rng.choice(N, args.pairs, replace=False))
    modes = ["baseline"] + [f"tau{t}" for t in taus]
    totals = {m: {"seg_errors": 0, "pose_sse": 0.0, "benefit": 0, "harm": 0, "wash": 0, "changed_px": 0} for m in modes}
    scored = 0
    started = time.perf_counter()
    receipt = out / f"AMPLITUDE_{args.source}_seed{args.seed}_n{args.pairs}.json"
    # Resume from disk: the receipt carries the running totals and the pairs done.
    done: list[int] = []
    if receipt.is_file():
        prior = json.loads(receipt.read_text())
        if prior.get("taus") == taus and prior.get("source") == args.source and prior.get("pairs") == pairs:
            done = list(prior["pairs_done"])
            scored = int(prior["pairs_scored"])
            for mode in modes:
                for key in totals[mode]:
                    totals[mode][key] = prior["rows"][mode][key]
    for pair in pairs:
        if pair in done:
            continue
        start = (pair // CHUNK) * CHUNK
        directory = ROOT / "chunks" / f"{start:03d}_{start + CHUNK:03d}"
        local = pair - start
        base_pair = np.array(np.load(directory / "baseline_camera_u8.npy", mmap_mode="r", allow_pickle=False)[local])
        treated = np.array(np.load(directory / f"{args.source}_camera_u8.npy", mmap_mode="r", allow_pickle=False)[local, 1])
        target = labels[pair]
        predictions = {}
        for mode in modes:
            frames = base_pair.copy()
            if mode != "baseline":
                frames[1] = clamp_delta(base_pair[1], treated, int(mode[3:]))
            totals[mode]["changed_px"] += int(np.count_nonzero((frames[1] != base_pair[1]).any(axis=-1)))
            with torch.inference_mode():
                pose, seg = net(torch.from_numpy(frames[None]))
            predictions[mode] = seg.argmax(dim=1)[0].numpy().astype(np.uint8)
            totals[mode]["pose_sse"] += float(np.square(pose["pose"][0, :6].numpy().astype(np.float64) - poses[pair]).sum())
            totals[mode]["seg_errors"] += int(np.count_nonzero(predictions[mode] != target))
        reference = predictions["baseline"]
        for mode in modes:
            changed = predictions[mode] != reference
            totals[mode]["benefit"] += int(np.count_nonzero(changed & (reference != target) & (predictions[mode] == target)))
            totals[mode]["harm"] += int(np.count_nonzero(changed & (reference == target) & (predictions[mode] != target)))
            totals[mode]["wash"] += int(np.count_nonzero(changed & (reference != target) & (predictions[mode] != target)))
        scored += 1
        done.append(pair)
        positions = scored * H * W
        rows = {}
        for mode in modes:
            value = dict(totals[mode])
            value["d_seg"] = value["seg_errors"] / positions
            value["d_pose"] = value["pose_sse"] / (scored * 6)
            value["delta_S_seg"] = 100 * (value["d_seg"] - totals["baseline"]["seg_errors"] / positions)
            value["delta_S_pose"] = (5 / np.sqrt(10 * POINTER_D_POSE)) * (value["d_pose"] - totals["baseline"]["pose_sse"] / (scored * 6))
            value["delta_S_local_linear"] = value["delta_S_seg"] + value["delta_S_pose"]
            rows[mode] = value
        payload = {"schema": "ddm_rbf1_amplitude.v1", "axis": "[macOS-CPU advisory]", "score_claim": False,
                   "scope": f"SEEDED RANDOM SUBSET of {args.pairs} pairs -- SCOPE reduction, no n600 verdict",
                   "source": args.source, "taus": taus, "seed": args.seed, "pairs": pairs,
                   "pairs_scored": scored, "pairs_done": sorted(done), "pointer_archive_bytes": POINTER_BYTES,
                   "host": platform.platform(), "threads": args.threads,
                   "elapsed_s": time.perf_counter() - started,
                   "pose_linearisation": "delta_S_pose uses the pointer's d_pose = 4.59e-06 operating point; it is a local linearisation, not the exact sqrt term",
                   "rows": rows}
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(receipt)
        print(json.dumps({"pairs_scored": scored, "pair": pair,
                          "delta_S": {m: round(rows[m]["delta_S_local_linear"], 8) for m in modes if m != "baseline"}}), flush=True)
    print(json.dumps({"receipt": str(receipt)}))


if __name__ == "__main__":
    main()
