"""ORDER ntb2's 14 retained 3-bit packets by how far each moves the RENDER.

This is a SCREEN, not a verdict. It renders a handful of pairs through each packet's
own archive and reports the pixel distance from the control render of the same pairs.
It answers only "which layers are even worth the n600 scorer", and no row here may be
cited as a d_seg, d_pose, or score result: `ddm_ntb2_renderer_score.py` at n600 is the
only instrument allowed to produce those.

The pixel distance IS on the receiver's own forward model (`jg1.render_frame1`, batch 1)
against the same token field, so the ordering is a property of the real render path and
not of a surrogate -- but a small pair count cannot estimate a population mean, and the
per-pair numbers are reported so a reader can see the spread rather than a single mean.

Axis: [macOS-CPU advisory, jg1 instrument] -- SCREEN ONLY. score_claim=false.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "src")]

import ddm_jg1_seg_solve as jg1
import numpy as np

from experiments import ddm_ntb2_control as control
from experiments import ddm_ntb2_renderer_score as score

AXIS = "[macOS-CPU advisory, jg1 instrument] SCREEN ONLY"


def forward_model_control(args) -> int:
    """Re-render move 44's own tokens through its own renderer and diff the SHIPPED decode.

    Without this, "the renderer moved the render by N levels" is a statement about a
    look-alike forward model. `jg1.forward_model_control` is the receiver's own path and
    the shipped `0.raw` is what the public decode actually produced.
    """
    import torch

    torch.set_num_threads(args.threads)
    tree = control.ROOT / "source_runtime"
    base_sha = score.sha256_file(tree / "archive.zip")
    if base_sha != control.BASE_SHA:
        raise SystemExit(f"base runtime archive moved: {base_sha}")
    rng = np.random.default_rng(args.seed)
    pairs = np.sort(rng.choice(jg1.N_PAIRS, size=args.pairs, replace=False)).astype(np.int64)
    tokens = jg1.load_tokens(score.TOKEN_FIELD)
    semantic = jg1.load_semantic_renderer(
        archive_path=tree / "archive.zip", runtime_dir=tree / "runtime"
    )
    verdict = jg1.forward_model_control(
        tokens,
        pairs,
        semantic=semantic,
        raw_path=args.forward_model_control,
        verify_raw_sha=False,
    )
    result = {
        "schema": "ddm_ntb2_forward_model_control.v1",
        "axis": AXIS.replace("SCREEN ONLY", "INSTRUMENT FALSIFIER"),
        "score_claim": False,
        "base_archive_sha256": base_sha,
        "shipped_raw": score.fact(args.forward_model_control),
        "token_field": score.fact(score.TOKEN_FIELD),
        "pairs": [int(p) for p in pairs],
        "producer": score.fact(Path(__file__)),
        "verdict": verdict,
        "passed": verdict["pixels_changed"] == 0,
    }
    score.retain(score.STORE / "FORWARD_MODEL_CONTROL.json", (json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
    print(json.dumps(result, indent=1, sort_keys=True), flush=True)
    return 0 if result["passed"] else 3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--out", type=Path, default=score.STORE / "SCREEN.json")
    parser.add_argument(
        "--forward-model-control",
        type=Path,
        default=None,
        help=(
            "instead of screening, ask the falsifier that makes every renderer row on this "
            "arm mean something: does re-rendering move 44's OWN tokens through move 44's "
            "OWN renderer reproduce the frames the shipped receiver actually decoded, byte "
            "for byte? Pass move 44's retained public 0.raw."
        ),
    )
    args = parser.parse_args(argv)
    if args.forward_model_control is not None:
        return forward_model_control(args)

    import torch

    torch.set_num_threads(args.threads)
    tree = control.ROOT / "source_runtime"
    base_sha = score.sha256_file(tree / "archive.zip")
    if base_sha != control.BASE_SHA:
        raise SystemExit(f"base runtime archive moved: {base_sha}")

    # A seeded RANDOM sample, never a prefix: the first pairs of this video are the two
    # hardest 60-pair blocks on the pose axis, so a prefix is a different population.
    rng = np.random.default_rng(args.seed)
    pairs = np.sort(rng.choice(jg1.N_PAIRS, size=args.pairs, replace=False)).astype(np.int64)
    tokens = jg1.load_tokens(score.TOKEN_FIELD)

    def render(archive: Path) -> np.ndarray:
        semantic = jg1.load_semantic_renderer(
            archive_path=archive, runtime_dir=tree / "runtime"
        )
        return jg1.render_frame1(semantic, tokens[pairs], pairs)

    started = time.time()
    reference = render(tree / "archive.zip")
    rows = []
    for packet in sorted(p for p in score.SEEDS.iterdir() if p.is_dir()):
        archive = packet / "archive.twin0.zip"
        if not archive.is_file() or packet.name in {"control", "runtime_copy"}:
            continue
        frames = render(archive)
        delta = np.abs(frames.astype(np.int16) - reference.astype(np.int16))
        rows.append(
            {
                "layer": packet.name,
                "archive_bytes": archive.stat().st_size,
                "delta_bytes_vs_move44": archive.stat().st_size - score.BASE_BYTES,
                "mean_abs_pixel_delta": float(delta.mean()),
                "max_abs_pixel_delta": int(delta.max()),
                "pixels_changed_fraction": float((delta > 0).mean()),
                "per_pair_mean_abs": [float(delta[i].mean()) for i in range(len(pairs))],
            }
        )
        print(json.dumps(rows[-1] | {"per_pair_mean_abs": "..."}, sort_keys=True), flush=True)

    rows.sort(key=lambda r: r["mean_abs_pixel_delta"])
    result = {
        "schema": "ddm_ntb2_renderer_screen.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "status": "SCREEN ONLY -- an ordering, never a d_seg/d_pose/score row",
        "base_archive_sha256": base_sha,
        "pairs": [int(p) for p in pairs],
        "pair_selection": f"seeded random sample, seed {args.seed}, never a prefix",
        "producer": score.fact(Path(__file__)),
        "elapsed_seconds": time.time() - started,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    score.retain(args.out, (json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
    print(json.dumps({"ordering": [(r["layer"], round(r["mean_abs_pixel_delta"], 4)) for r in rows]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
