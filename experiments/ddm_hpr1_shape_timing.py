"""ddm_hpr1 -- decode-time cost of a conv_past dilation change, measured not argued.

The HPAC loop sits inside the 1,232.42 s decode leg and the strict slack is 27.581 s,
about 2.2 %.  A shape rung that costs more time than that is dead however many bytes it
saves, so the rung's time must be measured before it is proposed.

WHAT IS TIMED.  ``conv_past`` is the only module a ``--past-dilation`` rung moves.  It
runs ONCE per frame, densely, over the previous plane's one-hot -- it is not inside the
per-group loop -- so the whole decode-time delta of the rung is the delta of this one
call, repeated 600 times.  The two dilations are timed INTERLEAVED (A/B/A/B) on the same
host in the same process so host drift cancels instead of accumulating into one arm.

The shipped module is constructed by the shipping loader from move 45's own bytes; only
``dilation`` and ``padding`` move between arms, exactly as the rung moves them.

Axis ``[macOS-CPU advisory; measured wall-clock, same-host interleaved]``;
``score_claim=false``.  A projection over 600 frames is a PROJECTION and is labelled as
one: the authority for a decode budget is a full public decode.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from experiments.ddm_hpr1_shape_price import FIELD, FIELD_SHA, POINTER45_SHA, PROMOTED45, fact

ARM_STORES = ("/Volumes/VertigoDataTier/pact/ddm_hpr1", "/Volumes/APDataStore/pact/ddm_hpr1")
FRAMES, HEIGHT, WIDTH, CLASSES = 600, 384, 512, 1
SYMBOLS_PER_FRAME = 384 * 512
STRICT_SLACK_SECONDS = 27.581


class TimingError(RuntimeError):
    """A timing input is not the shipped object."""


def measure(dilations: tuple[int, ...], repeats: int, frames: int) -> dict:
    import torch
    import torch.nn.functional as functional

    from experiments import ddm_rlc1_run as landed

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if fact(FIELD)["sha256"] != FIELD_SHA:
        raise TimingError("field sha mismatch")
    if fact(PROMOTED45 / "archive.zip")["sha256"] != POINTER45_SHA:
        raise TimingError("promoted tree is not the move-45 archive")
    rx, renderer, _ = landed.io.load_runtime(PROMOTED45)
    parts = rx.read_residual_archive(PROMOTED45 / "archive.zip")
    model = renderer.load_hpac(rx.materialize_ihs1(parts.hpac_blob, renderer), torch.device("cpu"))
    module = model.conv_past
    kernel = int(module.weight.shape[-1])

    field = np.memmap(FIELD, dtype=np.uint8, mode="r", shape=(FRAMES, HEIGHT, WIDTH))
    planes = torch.from_numpy(np.ascontiguousarray(field[:frames])).long()
    one_hot = functional.one_hot(planes, num_classes=model.num_classes).permute(0, 3, 1, 2).float()

    samples: dict[int, list[float]] = {dilation: [] for dilation in dilations}
    with torch.no_grad():
        for _ in range(2):  # warm the allocator; discarded
            for dilation in dilations:
                module.dilation, module.padding = dilation, dilation * (kernel - 1) // 2
                module(one_hot[:1])
        for _ in range(repeats):
            # Interleaved so host drift lands on both arms equally.
            for dilation in dilations:
                module.dilation, module.padding = dilation, dilation * (kernel - 1) // 2
                start = time.perf_counter()
                for index in range(frames):
                    module(one_hot[index : index + 1])
                samples[dilation].append(time.perf_counter() - start)

    rows = []
    for dilation in dilations:
        values = samples[dilation]
        per_frame = statistics.median(values) / frames
        rows.append(
            {
                "dilation": dilation,
                "median_seconds_for_sampled_frames": statistics.median(values),
                "seconds_per_frame": per_frame,
                "ns_per_symbol": per_frame / SYMBOLS_PER_FRAME * 1e9,
                "projected_seconds_over_600_frames": per_frame * FRAMES,
                "samples": values,
            }
        )
    base = next(row for row in rows if row["dilation"] == dilations[0])
    for row in rows:
        row["projected_delta_seconds_vs_first"] = (
            row["projected_seconds_over_600_frames"] - base["projected_seconds_over_600_frames"]
        )
        row["fraction_of_strict_slack"] = row["projected_delta_seconds_vs_first"] / STRICT_SLACK_SECONDS
    return {
        "module": "conv_past",
        "kernel": kernel,
        "frames_sampled": frames,
        "repeats": repeats,
        "strict_slack_seconds": STRICT_SLACK_SECONDS,
        "rows": rows,
        "note": (
            "conv_past runs once per frame outside the per-group loop, so this delta is "
            "the whole decode-time delta of a --past-dilation rung.  The 600-frame number "
            "is a PROJECTION, not a decode-budget authority."
        ),
        "axis": "[macOS-CPU advisory; measured wall-clock, same-host interleaved]",
        "score_claim": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--frames", type=int, default=40)
    args = parser.parse_args()
    if not any(str(args.out_dir).startswith(root) for root in ARM_STORES):
        raise TimingError("out-dir must be inside this arm's store")
    payload = measure(tuple(args.dilations), args.repeats, args.frames)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "TIMING_conv_past.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**payload, "rows": [{k: v for k, v in r.items() if k != "samples"} for r in payload["rows"]]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
