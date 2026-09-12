"""ddm_ren2 deliverable 3: the AMPLITUDE CURVE over a refit's checkpoint ladder.

The bound this builds is ``ddm_ren1`` §4.6 item 2: *"Nothing in the trainer constrains
the render delta to stay inside the carrier's capacity, and that constraint is the whole
content of the finding."*  ren1 measured the carrier's pose absorption AT 0.5 LSB camera
RMS; a checkpoint above that amplitude is outside the regime the absorption number was
measured in, and pricing it on that number would be the borrowed-number genus.

For every checkpoint it reports, all MEASURED on the REALIZED (packed-and-parsed) state:

* ``realized_values_changed`` -- how many of the 66,339 stored values the deployed
  quantizer actually moved.  This is the ladder's real x-axis: ``ddm_rw1`` measured that
  the shipped grid's SMALLEST action is one int4 code step and that it already moves
  240-455 argmax cells, so the refit's action is coarse by construction and a checkpoint
  with zero changed values is a no-op wearing a training run's name.
* ``camera_lsb_rms`` with ``ddm_ren1``'s smooth/noise split.
* ``member_delta_bytes`` by the REAL encode.

Seeded RANDOM pair sampling via ``up2.select_pairs``, never a prefix.

Axis: ``[macOS-CPU advisory, jg1 instrument]``.  No score claim, no promotion.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "src")]

import ddm_jg1_seg_solve as jg1
import ddm_ren2_price_checkpoint as price
import ddm_ren2_restore_init as ren2_init
import ddm_up2_shipping_pose_solve as up2
import numpy as np

STORE = Path("/Volumes/VertigoDataTier/pact/ddm_ren2/amplitude")


def measure(checkpoint: Path, *, shipped, tokens, shipped_semantic, shipped_state, pairs):
    state, provenance = price.load_candidate_state(checkpoint)
    realized = price.realize(state, shipped=shipped)
    encoded = realized["encoded"]
    realized_changed = int(
        sum(
            int((encoded["realized_state"][name] != shipped_state[name]).sum())
            for name in shipped_state
        )
    )
    float_changed = int(
        sum(int((state[name] != shipped_state[name]).sum()) for name in shipped_state)
    )
    per_tensor = {
        name: int((encoded["realized_state"][name] != shipped_state[name]).sum())
        for name in shipped_state
        if int((encoded["realized_state"][name] != shipped_state[name]).sum())
    }
    semantic = price.renderer_with_state(shipped, realized["parsed"])
    totals, smooths, noises, changed = [], [], [], []
    frame_max = 0.0
    for pair in pairs:
        chunk = np.array([int(pair)], dtype=np.int64)
        base = jg1.render_frame1(shipped_semantic, tokens[chunk], chunk)[0]
        cand = jg1.render_frame1(semantic, tokens[chunk], chunk)[0]
        delta = cand.astype(np.float64) - base.astype(np.float64)
        total, smooth, noise = price.smooth_noise_split(delta)
        totals.append(total)
        smooths.append(smooth)
        noises.append(noise)
        frame_max = max(frame_max, float(np.abs(delta).max()))
        changed.append(int(np.count_nonzero(cand != base)))
    rms = float(np.sqrt(np.mean(np.square(totals))))
    smooth_rms = float(np.sqrt(np.mean(np.square(smooths))))
    noise_rms = float(np.sqrt(np.mean(np.square(noises))))
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": ren2_init.fact(checkpoint)["sha256"],
        "step": provenance["step"],
        "phase": provenance["phase"],
        "float_values_changed": float_changed,
        "realized_values_changed": realized_changed,
        "realized_changed_by_tensor": per_tensor,
        "member_bytes": encoded["member_bytes"],
        "member_delta_bytes": encoded["member_bytes"] - shipped["member_bytes"],
        "pairs": len(pairs),
        "camera_lsb_rms": rms,
        "smooth_band_rms": smooth_rms,
        "noise_band_rms": noise_rms,
        "smooth_over_noise": smooth_rms / noise_rms if noise_rms > 0 else None,
        "dominant_band": "smooth" if smooth_rms >= noise_rms else "noise",
        "max_abs_lsb": frame_max,
        "mean_camera_fraction_changed": float(np.mean(changed))
        / (jg1.CAMERA_H * jg1.CAMERA_W * 3),
        "inside_ceiling": bool(rms <= price.AMPLITUDE_CEILING_LSB),
    }


def run(args) -> int:
    started = time.time()
    out = STORE / args.label
    out.mkdir(parents=True, exist_ok=True)
    pointer = ren2_init.bind_pointer()
    shipped = price.load_shipped()
    tokens = jg1.load_tokens(ren2_init.TOKEN_FIELD)
    shipped_semantic = jg1.load_semantic_renderer(
        archive_path=ren2_init.POINTER_TREE / "archive.zip",
        runtime_dir=ren2_init.POINTER_TREE / "runtime",
    )
    shipped_state = {
        name: value.detach().cpu().clone()
        for name, value in shipped_semantic.state_dict().items()
    }
    pairs = up2.select_pairs(args.pairs, args.seed)

    checkpoints: list[Path] = []
    for pattern in args.checkpoints:
        if "*" in pattern:
            # Path().glob() refuses an ABSOLUTE pattern ("Non-relative patterns are
            # unsupported"), and every path this arm uses is absolute -- so glob the
            # parent by name instead of handing the whole path to glob().
            candidate = Path(pattern)
            matched = sorted(candidate.parent.glob(candidate.name))
        else:
            matched = [Path(pattern)]
        checkpoints.extend(path for path in matched if path.is_file())
    if not checkpoints:
        raise price.Ren2PriceError("no checkpoints matched")

    rows: list[dict[str, Any]] = []
    rows_path = out / "curve.jsonl"
    with rows_path.open("a", encoding="utf-8") as stream:
        for checkpoint in checkpoints:
            row = measure(
                checkpoint,
                shipped=shipped,
                tokens=tokens,
                shipped_semantic=shipped_semantic,
                shipped_state=shipped_state,
                pairs=pairs,
            )
            rows.append(row)
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "step",
                            "realized_values_changed",
                            "camera_lsb_rms",
                            "smooth_band_rms",
                            "noise_band_rms",
                            "member_delta_bytes",
                            "inside_ceiling",
                        )
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    result = {
        "schema": "ddm_ren2_amplitude_curve.v1",
        "axis": "[macOS-CPU advisory, jg1 instrument]",
        "score_claim": False,
        "promotion_eligible": False,
        "label": args.label,
        "pointer": pointer,
        "pointer_move": 48,
        "sample": {
            "pairs": len(pairs),
            "seed": args.seed,
            "shape": "seeded random via up2.select_pairs",
        },
        "ceiling_lsb": price.AMPLITUDE_CEILING_LSB,
        "rows": rows,
        "producer": ren2_init.fact(Path(__file__)),
        "elapsed_seconds": time.time() - started,
    }
    path = out / "RESULT.json"
    payload = (json.dumps(result, indent=1, sort_keys=True) + "\n").encode()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)
    print(json.dumps({"rows": len(rows), "out": str(path)}), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--threads", type=int, default=3)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    STORE.mkdir(parents=True, exist_ok=True)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
