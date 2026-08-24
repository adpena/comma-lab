#!/usr/bin/env python3
"""ddm_tv1 -- capture the SHIPPED coder's per-position conditional distribution.

WHY THIS EXISTS
---------------
``ddm_tv1`` measures how far the dx2 token field can move before the frozen
scorer notices.  "Move" is a family, not a single thing, and the choice IS the
mechanism: a uniform-over-alphabet reassignment is a maximal perturbation in a
field whose bit mass has Gini 0.9951, so it would measure receiver robustness
to garbage rather than the size of the scorer's equivalence cell.  The
reference form is a resample from the field's OWN conditional -- the
distribution the shipped coder actually prices each position against.

That distribution is computed inside ``residual_archive.decode_production_tokens``
and then thrown away: only two SHA-256 digests of it survive into the token
checkpoint receipt.  This module replays the exact same model/corrector loop and
RETAINS the distribution, feeding the ALREADY-DECODED tokens as the symbol
stream instead of pulling them from the RC64 decoder.

Substituting known symbols for decoded symbols is sound because the loop is
causal in exactly one quantity: the decoded token plane.  ``previous`` /
``current`` / ``corrector.observe`` / ``corrector.end_frame`` all consume
symbols, and the symbols we feed are the very ones the shipped decoder produced
(the retained field's SHA-256 equals the checkpoint receipt's
``decoded_token_sha256``).  The claim is not asserted: the replay recomputes
``corrected_quantized_logit_sha256`` and ``corrected_cdf_input_sha256`` and
REFUSES unless both equal the shipped receipt.  A replay that drifted anywhere
in the model, the residual table, the boundary buckets, or the corrector state
cannot reproduce those digests.

NOT A DECODER.  This module never advances an arithmetic decoder and never
produces an archive.  It reads a retained token field and writes a probability
field.  It cannot be used to code anything.
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

NUM_CLASSES = 5


class CaptureError(RuntimeError):
    """Fail-closed error for the conditional capture."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runtime(runtime_root: Path):
    """Import the pinned candidate runtime as the package ``runtime``."""
    runtime_root = runtime_root.resolve()
    if not (runtime_root / "runtime" / "f26_inflate.py").is_file():
        raise CaptureError(f"no runtime/f26_inflate.py under {runtime_root}")
    sys.path.insert(0, str(runtime_root))
    import runtime.f26_inflate as f26_inflate
    import runtime.residual_archive as residual_archive

    return f26_inflate, residual_archive


def capture(
    *,
    runtime_root: Path,
    archive_path: Path,
    tokens_path: Path,
    receipt_path: Path,
    out_path: Path,
    threads: int,
) -> dict[str, Any]:
    f26_inflate, residual_archive = load_runtime(runtime_root)
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = receipt["token_decoder"]
    if sha256_of(tokens_path) != expected["decoded_token_sha256"]:
        raise CaptureError("token field does not match the checkpoint receipt")

    parts = residual_archive.read_residual_archive(archive_path)
    if parts.table is None:
        raise CaptureError("archive carries no residual correction table")
    renderer = f26_inflate._load_renderer(runtime_root / "cpr1")
    plane = int(renderer.EVAL_H) * int(renderer.EVAL_W)
    frames = int(renderer.N)

    tokens = np.fromfile(tokens_path, dtype=np.uint8).reshape(
        frames, int(renderer.EVAL_H), int(renderer.EVAL_W)
    )

    device = torch.device("cpu")
    base_hpac = residual_archive.materialize_ihs1(parts.hpac_blob, renderer)
    model = renderer.load_hpac(base_hpac, device)
    masks = renderer.group_masks(device)
    sparse = residual_archive._sparse_class(runtime_root / "cpr1")(
        model, renderer.EVAL_H, renderer.EVAL_W
    )
    corrector = residual_archive._rr8_select_corrector(plane)
    corrector_kind = residual_archive._rr8_corrector_kind(corrector)
    if corrector_kind != expected["free_corrector"]:
        raise CaptureError(
            f"corrector {corrector_kind!r} differs from the shipped "
            f"{expected['free_corrector']!r}; the replay would price a different field"
        )

    group_plans = []
    for mask in masks:
        flat = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1))
        group_plans.append((torch.from_numpy(flat).to(device), flat))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    conditional = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(frames * plane, NUM_CLASSES)
    )

    corrected_digest = hashlib.sha256()
    cdf_digest = hashlib.sha256()
    started = time.time()
    with torch.inference_mode():
        from runtime.hpac_inference import optimize_sparse_evaluator

        optimize_sparse_evaluator(sparse)
        previous = torch.zeros((1, renderer.EVAL_H, renderer.EVAL_W), dtype=torch.long)
        for frame in range(frames):
            index = torch.tensor([frame], dtype=torch.long, device=device)
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(index, previous)
            if frame:
                previous_cpu = previous[0].to(dtype=torch.uint8).numpy()
                boundary = residual_archive._boundary_buckets(previous_cpu).reshape(-1)
            else:
                boundary = np.full(plane, 4, dtype=np.uint8)
            corrector.begin_frame(boundary)
            truth = tokens[frame].reshape(-1)
            for group, (device_positions, flat_positions) in enumerate(group_plans):
                selected = sparse.selected_logits(current, context, group)
                base_logits = selected.cpu().numpy()
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                feature = (
                    boundary[flat_positions].astype(np.int64) * NUM_CLASSES + predicted
                )
                corrected = base_logits + parts.table.values[feature]
                corrected_digest.update(
                    np.ascontiguousarray(corrected, dtype="<f4").tobytes()
                )
                probability = residual_archive._probability_table(
                    corrected, renderer.HPAC_LOGIT_PRECISION
                )
                cdf_digest.update(
                    np.ascontiguousarray(probability, dtype="<f4").tobytes()
                )
                state = corrector.group_state(probability, predicted, flat_positions)
                conditional[frame * plane + flat_positions] = corrector.coding_row(state)
                symbols = truth[flat_positions].astype(np.int64)
                corrector.observe(state, symbols)
                current.reshape(-1)[device_positions] = torch.from_numpy(symbols)
            if not np.array_equal(
                current[0].to(dtype=torch.uint8).numpy().reshape(-1), truth
            ):
                raise CaptureError(f"frame {frame} plane did not reassemble the field")
            corrector.end_frame(truth)
            previous = current
    elapsed = time.time() - started
    conditional.flush()

    got_corrected = corrected_digest.hexdigest()
    got_cdf = cdf_digest.hexdigest()
    if got_corrected != expected["corrected_quantized_logit_sha256"]:
        raise CaptureError("replayed corrected-logit digest differs from the shipped run")
    if got_cdf != expected["corrected_cdf_input_sha256"]:
        raise CaptureError("replayed CDF-input digest differs from the shipped run")

    del conditional
    return {
        "schema": "ddm_tv1_coding_conditional_capture.v1",
        "runtime_root": str(runtime_root),
        "archive_sha256": sha256_of(archive_path),
        "tokens_sha256": expected["decoded_token_sha256"],
        "free_corrector": corrector_kind,
        "replay_seconds": elapsed,
        "digest_control": {
            "corrected_quantized_logit_sha256": got_corrected,
            "corrected_cdf_input_sha256": got_cdf,
            "matches_shipped_receipt": True,
        },
        "conditional": {
            "path": str(out_path),
            "bytes": out_path.stat().st_size,
            "sha256": sha256_of(out_path),
            "shape": [frames * plane, NUM_CLASSES],
            "dtype": "float32",
            "semantics": "shipped corrector coding row p(class | context) per position",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(args.threads)

    report = capture(
        runtime_root=args.runtime_root,
        archive_path=args.archive,
        tokens_path=args.tokens,
        receipt_path=args.receipt,
        out_path=args.out,
        threads=args.threads,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
