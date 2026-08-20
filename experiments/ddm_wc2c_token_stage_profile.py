#!/usr/bin/env python3
"""ddm_wc2c - per-step profile of the shipping F26 token decode loop.

WHY THIS EXISTS.  ``ddm_wc2`` measured the jg5 token stage at 1,341.540 s on the
T4 -- 95.72% of a 1,419.900 s inflate that REFUSES the CI wall.  That number is a
single opaque total over a 114,000-iteration loop
(``runtime/residual_archive.py:611-644``).  Nothing splits it.  The port decision
-- how much of the loop must be lowered into C, and therefore how much of the
float64 corrector stack must be re-implemented under IEEE identity discipline --
depends entirely on which of the twelve per-iteration steps holds the seconds.

WHAT IT MEASURES.  A frame-prefix decode of the real archive with a
``time.perf_counter`` around each step of the real loop.  The loop body is a
verbatim transcription of ``decode_production_tokens``; the only additions are
the timers and the prefix bound.  Every decoded value, digest, and RC64 bit
position is therefore the shipping value for that prefix.

WHY A PREFIX IS LEGITIMATE HERE.  RC64 is one sequential stream decoded from the
start and the model is autoregressive over frames, so frames ``[0, n)`` decode to
exactly the bytes the full run produces for those frames.  A prefix is a valid
SUFFIX-TRUNCATED run, not a subsample.  It is NOT a valid basis for a population
claim (m88: a prefix of a skewed population is a different population), so this
probe reports per-step SHARES and a per-iteration cost, and labels every
extrapolation to n600 as a projection.

AXIS.  ``[macOS-CPU advisory]`` / ``[macOS-MPS advisory]``.  Never a score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CANDIDATE = Path(
    "/Volumes/APDataStore/pact/ddm_jg5/candidate_runtime_jg5"
)

STEP_NAMES = (
    "frame_context",
    "boundary_buckets",
    "sparse_selected_logits",
    "device_to_host",
    "argmax_and_table",
    "corrected_digest",
    "probability_table",
    "cdf_digest",
    "corrector_group_state",
    "corrector_coding_row",
    "rc64_decode",
    "corrector_observe",
    "host_to_device",
    "frame_epilogue",
)


class Wc2cProfileError(RuntimeError):
    """A precondition of the profile run failed."""


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _import_candidate(candidate: Path):
    """Import the candidate runtime tree as the ``runtime`` package."""
    candidate = candidate.resolve()
    if not (candidate / "runtime" / "f26_inflate.py").is_file():
        raise Wc2cProfileError(f"no candidate runtime tree at {candidate}")
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
    import runtime.f26_inflate as f26_inflate
    import runtime.residual_archive as residual_archive

    return residual_archive, f26_inflate


def profile_token_decode(
    candidate: Path,
    frames: int,
    device_name: str,
    num_threads: int,
) -> dict[str, Any]:
    """Run the shipping token loop over a frame prefix with per-step timers."""
    residual_archive, f26_inflate = _import_candidate(candidate)
    import torch
    from runtime.free_corrector import FreeCorrector
    from runtime.hpac_inference import (
        configure_cuda_reproducibility,
        optimize_sparse_evaluator,
    )

    torch.set_num_threads(int(num_threads))
    device = torch.device(device_name)

    archive_path = candidate / "archive.zip"
    renderer_dir = candidate / "cpr1"
    parts = residual_archive.read_residual_archive(archive_path)
    if parts.table is None:
        raise Wc2cProfileError("archive carries no residual correction table")
    renderer = f26_inflate._load_renderer(renderer_dir)

    if device.type == "cuda":
        configure_cuda_reproducibility()
    base_hpac = residual_archive.materialize_ihs1(parts.hpac_blob, renderer)
    model = renderer.load_hpac(base_hpac, device)
    masks = renderer.group_masks(device)
    sparse = residual_archive._sparse_class(renderer_dir)(
        model, renderer.EVAL_H, renderer.EVAL_W
    )
    corrector = FreeCorrector(renderer.EVAL_H * renderer.EVAL_W)

    library = os.environ.get("CPR1_RC64_LIBRARY")
    if not library:
        raise Wc2cProfileError("CPR1_RC64_LIBRARY is required")
    decoder = residual_archive.NativeDecoder(Path(library), parts.token_stream)

    group_plans = []
    for mask in masks:
        mask_array = mask.detach().cpu().numpy()
        flat_positions = np.flatnonzero(mask_array.reshape(-1))
        group_plans.append(
            (torch.from_numpy(flat_positions).to(device), flat_positions)
        )

    total_frames = min(int(frames), int(renderer.N))
    seconds = dict.fromkeys(STEP_NAMES, 0.0)
    iterations = 0
    clock = time.perf_counter

    corrected_digest = hashlib.sha256()
    cdf_digest = hashlib.sha256()
    started = clock()
    with torch.inference_mode():
        optimize_sparse_evaluator(sparse)
        previous = torch.zeros(
            (1, renderer.EVAL_H, renderer.EVAL_W), dtype=torch.long, device=device
        )
        tokens = torch.empty(
            (total_frames, renderer.EVAL_H, renderer.EVAL_W), dtype=torch.uint8
        )
        for frame in range(total_frames):
            mark = clock()
            index = torch.tensor([frame], dtype=torch.long, device=device)
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(index, previous)
            seconds["frame_context"] += clock() - mark

            mark = clock()
            if frame:
                previous_cpu = (
                    previous[0].to(device="cpu", dtype=torch.uint8).numpy()
                )
                boundary = residual_archive._boundary_buckets(
                    previous_cpu
                ).reshape(-1)
            else:
                boundary = np.full(
                    renderer.EVAL_H * renderer.EVAL_W, 4, dtype=np.uint8
                )
            corrector.begin_frame(boundary)
            seconds["boundary_buckets"] += clock() - mark

            for group, (device_positions, flat_positions) in enumerate(group_plans):
                mark = clock()
                selected = sparse.selected_logits(current, context, group)
                seconds["sparse_selected_logits"] += clock() - mark

                mark = clock()
                base_logits = selected.cpu().numpy()
                seconds["device_to_host"] += clock() - mark

                mark = clock()
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                feature = (
                    boundary[flat_positions].astype(np.int64)
                    * residual_archive.NUM_CLASSES
                    + predicted
                )
                corrected = base_logits + parts.table.values[feature]
                seconds["argmax_and_table"] += clock() - mark

                mark = clock()
                corrected_digest.update(
                    np.ascontiguousarray(corrected, dtype="<f4").tobytes()
                )
                seconds["corrected_digest"] += clock() - mark

                mark = clock()
                probability = residual_archive._probability_table(
                    corrected, renderer.HPAC_LOGIT_PRECISION
                )
                seconds["probability_table"] += clock() - mark

                mark = clock()
                cdf_digest.update(
                    np.ascontiguousarray(probability, dtype="<f4").tobytes()
                )
                seconds["cdf_digest"] += clock() - mark

                mark = clock()
                state = corrector.group_state(
                    probability, predicted, flat_positions
                )
                seconds["corrector_group_state"] += clock() - mark

                mark = clock()
                coding_row = corrector.coding_row(state)
                seconds["corrector_coding_row"] += clock() - mark

                mark = clock()
                symbols = decoder.decode(coding_row).astype(np.int64)
                seconds["rc64_decode"] += clock() - mark

                mark = clock()
                corrector.observe(state, symbols)
                seconds["corrector_observe"] += clock() - mark

                mark = clock()
                current.reshape(-1)[device_positions] = torch.from_numpy(
                    symbols
                ).to(device)
                seconds["host_to_device"] += clock() - mark
                iterations += 1

            mark = clock()
            tokens[frame] = current[0].to(device="cpu", dtype=torch.uint8)
            corrector.end_frame(tokens[frame].numpy().reshape(-1))
            previous = current
            seconds["frame_epilogue"] += clock() - mark
    elapsed = clock() - started

    accounted = sum(seconds.values())
    ranked = sorted(seconds.items(), key=lambda item: -item[1])
    return {
        "axis": f"[macOS-{device_name.upper()} advisory]",
        "verdict_scope": "prefix-of-n600; per-step shares only, never a population claim",
        "candidate": str(candidate),
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "archive_bytes": archive_path.stat().st_size,
        "device": device_name,
        "torch_threads": int(num_threads),
        "frames": total_frames,
        "frames_total_in_archive": int(renderer.N),
        "groups_per_frame": len(group_plans),
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "accounted_seconds": accounted,
        "unaccounted_seconds": elapsed - accounted,
        "seconds_per_iteration": elapsed / iterations if iterations else None,
        "step_seconds": seconds,
        "step_share_of_elapsed": {
            name: (value / elapsed if elapsed else None)
            for name, value in seconds.items()
        },
        "ranked_steps": [{"step": name, "seconds": value} for name, value in ranked],
        "prefix_corrected_quantized_logit_sha256": corrected_digest.hexdigest(),
        "prefix_corrected_cdf_input_sha256": cdf_digest.hexdigest(),
        "prefix_decoded_token_sha256": hashlib.sha256(
            tokens.numpy().tobytes()
        ).hexdigest(),
        "decoder_bit_position": int(decoder.bit_position),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = profile_token_decode(
        candidate=args.candidate,
        frames=args.frames,
        device_name=args.device,
        num_threads=args.threads,
    )
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
