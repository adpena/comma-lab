"""ddm_ren2: the RANDOM-DIRECTION control at a refit's OWN measured amplitude.

The question this exists to answer, stated so it can come out either way: **did 1,500
steps of gradient descent buy anything a coin flip of the same size would not?**

``ddm_ren1`` measured that an iid half-LSB field costs +14 % ``d_seg``.  A refit that
lands its own render delta at amplitude A and pays the SAME damage a random field of
amplitude A pays has learned nothing about seg -- its descent is indistinguishable from
noise at the realized grid, and the axis is closed by the grid rather than by the
search.  A refit that pays materially LESS has found direction, and the remaining
question is only how much more of it there is.

``ddm_rw1`` ran a random-direction control in its own formulation; this is that control
lifted onto THIS formulation -- restored init, deployed mixed grid, current coded field,
realized through the shipped receiver -- and matched to the refit's MEASURED amplitude
rather than to a round number.

**The control matches the REALIZED delta, not a pre-round field, and that distinction is
load-bearing.**  A first draft scaled ``ddm_ren1``'s iid Gaussian to the refit's RMS and
handed it to the receiver's ``clamp(0,255).round()`` tail.  The camera raster is integer,
so a Gaussian of sigma 0.143 LSB leaves a sample unchanged unless ``|x| > 0.5``: the
realized delta came back at **0.0216 LSB RMS over 0.047 % of samples** instead of the
target 0.1430 over 2.04 % -- a 6.6x under-shoot that would have made the control look
harmless for a reason that has nothing to do with direction.  MEASURED, then fixed.

The refit's realized delta is a SPARSE +/-1 DITHER: ``max_abs_lsb = 1.0`` and
``fraction_changed = RMS^2`` to seven figures.  So the matched random direction is the
same object with random positions and signs -- identical amplitude, identical sparsity,
identical maximum, identical (noise) band, only the DIRECTION randomised.  It is applied
through ``ddm_ren1``'s own ``apply_perturbation`` so the receiver tail is the shipped one.

Axis: ``[macOS-CPU advisory, jg1 instrument, DALI GT lineage]``.  No score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "src")]

import ddm_jg1_seg_solve as jg1
import ddm_ren1_step0_probe as ren1_probe
import ddm_ren2_price_checkpoint as price
import ddm_ren2_restore_init as ren2_init
import ddm_up2_shipping_pose_solve as up2
import numpy as np

STORE = Path("/Volumes/VertigoDataTier/pact/ddm_ren2/random_control")
RESERVE_BYTES = 40 << 30


def retain(path: Path, payload: bytes) -> dict[str, Any]:
    if not path.resolve().is_relative_to(STORE.resolve()):
        raise price.Ren2PriceError(f"write outside the ren2 random-control store: {path}")
    if shutil.disk_usage(STORE.parent).free < RESERVE_BYTES + len(payload):
        raise price.Ren2PriceError("STORAGE_BLOCK: free space below reserve")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return ren2_init.fact(path)


def run(args) -> int:
    import torch

    started = time.time()
    out = STORE / args.label
    out.mkdir(parents=True, exist_ok=True)
    pointer = ren2_init.bind_pointer()
    tokens = jg1.load_tokens(ren2_init.TOKEN_FIELD)
    semantic = jg1.load_semantic_renderer(
        archive_path=ren2_init.POINTER_TREE / "archive.zip",
        runtime_dir=ren2_init.POINTER_TREE / "runtime",
    )
    net = jg1.load_segnet()
    gt_labels = jg1.load_gt_seg_labels(up2.LINEAGE_DALI)

    # The matched random direction: a sparse +/-1 dither whose density reproduces the
    # refit's REALIZED RMS (RMS^2 == changed fraction for a +/-1 dither, which is what
    # the refit's own amplitude leg measures to seven figures).  Built once and shared
    # by every pair, exactly as ren1 shares one field across its treatments, so the arm
    # is deterministic and pair-comparable.
    if args.spectrum != "noise":
        raise price.Ren2PriceError(
            "only the noise spectrum is a matched control for a sparse +/-1 dither; a "
            "smooth field cannot reproduce max_abs_lsb == 1 at this density"
        )
    density = float(args.rms_lsb) ** 2
    if not 0.0 < density < 1.0:
        raise price.Ren2PriceError("target RMS must be in (0, 1) LSB for a +/-1 dither")
    generator = torch.Generator().manual_seed(args.seed)
    shape = (jg1.CAMERA_H, jg1.CAMERA_W, 3)
    draw = torch.rand(shape, generator=generator)
    sign = torch.where(
        torch.rand(shape, generator=generator) < 0.5, -1.0, 1.0
    )
    field = torch.where(draw < density, sign, torch.zeros(()))
    scale = density

    shipped_argmax = None
    if price.REN1_CONTROL_PLANES.is_file():
        with np.load(price.REN1_CONTROL_PLANES) as blob:
            shipped_argmax = blob["argmax_control"]

    binding = {
        "schema": "ddm_ren2_random_direction_control.v1",
        "axis": "[macOS-CPU advisory, jg1 instrument, DALI GT lineage]",
        "score_claim": False,
        "promotion_eligible": False,
        "label": args.label,
        "pointer": pointer,
        "pointer_move": 48,
        "spectrum": args.spectrum,
        "target_rms_lsb": args.rms_lsb,
        "dither_density": scale,
        "construction": (
            "sparse +/-1 dither, iid positions at density RMS^2, iid signs; matches the "
            "refit's REALIZED delta (max_abs 1.0, fraction == RMS^2) rather than a "
            "pre-round field the receiver's round() would discard"
        ),
        "seed": args.seed,
        "producer": ren2_init.fact(Path(__file__)),
        "ren1_probe": ren2_init.fact(Path(ren1_probe.__file__)),
        "jg1": ren2_init.fact(Path(jg1.__file__)),
    }
    retain(out / "INPUTS.json", (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode())

    rows_path = out / "seg_rows.jsonl"
    done: dict[int, float] = {}
    realized_rms: list[float] = []
    changed: list[float] = []
    if args.resume and rows_path.is_file():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[int(row["pair"])] = float(row["d_seg"])
                realized_rms.append(float(row["realized_rms_lsb"]))
                changed.append(float(row["camera_fraction_changed"]))
    with rows_path.open("a", encoding="utf-8") as stream:
        for start in range(0, jg1.N_PAIRS, args.chunk):
            chunk = np.arange(start, min(start + args.chunk, jg1.N_PAIRS), dtype=np.int64)
            pending = np.array([p for p in chunk if int(p) not in done], dtype=np.int64)
            if pending.size == 0:
                continue
            base = jg1.render_frame1(semantic, tokens[pending], pending)
            moved = ren1_probe.apply_perturbation(base, field)
            argmax = jg1.argmax_from_camera_frames(net, moved)
            per_pair = jg1.d_seg_per_pair(argmax, gt_labels[pending])
            for row_index, (pair, value) in enumerate(zip(pending, per_pair, strict=True)):
                delta = moved[row_index].astype(np.float64) - base[row_index].astype(
                    np.float64
                )
                rms = float(np.sqrt(np.square(delta).mean()))
                fraction = float(
                    np.count_nonzero(moved[row_index] != base[row_index]) / delta.size
                )
                cells = (
                    int(np.count_nonzero(argmax[row_index] != shipped_argmax[int(pair)]))
                    if shipped_argmax is not None
                    else None
                )
                done[int(pair)] = float(value)
                realized_rms.append(rms)
                changed.append(fraction)
                stream.write(
                    json.dumps(
                        {
                            "pair": int(pair),
                            "d_seg": float(value),
                            "realized_rms_lsb": rms,
                            "camera_fraction_changed": fraction,
                            "cells_moved": cells,
                        }
                    )
                    + "\n"
                )
            stream.flush()
            print(
                json.dumps(
                    {
                        "done": len(done),
                        "of": jg1.N_PAIRS,
                        "running_mean": float(np.mean(list(done.values()))),
                        "seconds": round(time.time() - started, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    d_seg = float(np.mean([done[p] for p in range(jg1.N_PAIRS)]))
    result = {
        **binding,
        "pairs": jg1.N_PAIRS,
        "d_seg": d_seg,
        "realized_rms_lsb": float(np.sqrt(np.mean(np.square(realized_rms)))),
        "mean_camera_fraction_changed": float(np.mean(changed)),
        "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.time() - started,
    }
    retain(out / "RESULT.json", (json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
    print(json.dumps(result, indent=1, sort_keys=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--rms-lsb", required=True, type=float)
    parser.add_argument("--spectrum", default="noise", choices=("noise", "smooth"))
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
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
