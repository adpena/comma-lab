"""ddm_rc4 Stage 1 - exact rate-distortion structure of the hv1 token stream.

Replays the SHIPPED F26 token decoder forward with teacher forcing against the
RETAINED decoded token field of the live frontier archive, and accumulates the
exact per-position code length and model agreement.

POSITIVE CONTROL (instrument validity, fail-closed): the replay recomputes the
decoder's own `corrected_quantized_logit_sha256` and `corrected_cdf_input_sha256`
and refuses to emit a verdict unless both match the retained decode receipt.
If they match, every probability table here is bit-identical to the one the
shipping RC64 decoder consumed.

ALWAYS KEEP THE PAYLOAD: every materialized array is written to the SSD store
with sha256 + byte count.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

GEN = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pq1_submission_packet/generations/"
    "hv1_ep0634_s1p25_c1p0_brotli_q10"
)
STORE = Path("/Volumes/APDataStore/pact/ddm_rc4_rung4_token_drop_20260816")
RETAINED = STORE / "retained" / "token_rd"
TOKENS_U8 = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2/inflated/"
    ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
TOKENS_RECEIPT = TOKENS_U8.with_suffix(".json")

ARCHIVE_SHA = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
NUM_CLASSES = 5

# u = -log2(1 - p_max), the natural threshold coordinate near p_max -> 1.
U_MAX = 48.0
U_STEP = 0.125
U_BINS = int(U_MAX / U_STEP) + 2


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    RETAINED.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(GEN))
    sys.path.insert(0, str(GEN / "cpr1"))

    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    import importlib.util

    from runtime.hpac_inference import optimize_sparse_evaluator
    from runtime.ihs2 import materialize_ihs1
    from runtime.residual_archive import (
        _boundary_buckets,
        _probability_table,
        _sparse_class,
        read_residual_archive,
    )

    spec = importlib.util.spec_from_file_location("_rc4_renderer", GEN / "cpr1" / "inflate.py")
    runtime = importlib.util.module_from_spec(spec)
    sys.modules["_rc4_renderer"] = runtime
    spec.loader.exec_module(runtime)

    archive = GEN / "archive.zip"
    if sha256_bytes(archive.read_bytes()) != ARCHIVE_SHA:
        raise SystemExit("archive sha mismatch - refusing")
    parts = read_residual_archive(archive)
    token_stream_bytes = len(parts.token_stream)

    receipt = json.loads(TOKENS_RECEIPT.read_text())
    if receipt["binding"]["archive_sha256"] != ARCHIVE_SHA:
        raise SystemExit("retained token checkpoint is not bound to the hv1 archive")
    raw_tokens = np.fromfile(TOKENS_U8, dtype=np.uint8)
    if sha256_bytes(raw_tokens.tobytes()) != receipt["tokens"]["sha256"]:
        raise SystemExit("retained token payload sha mismatch")
    tokens = raw_tokens.reshape(runtime.N, runtime.EVAL_H, runtime.EVAL_W)

    device = torch.device("cpu")
    base_hpac = materialize_ihs1(parts.hpac_blob, runtime)
    model = runtime.load_hpac(base_hpac, device)
    masks = runtime.group_masks(device)
    sparse = _sparse_class(GEN / "cpr1")(model, runtime.EVAL_H, runtime.EVAL_W)

    group_plans = []
    for mask in masks:
        flat = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1))
        group_plans.append((torch.from_numpy(flat).to(device), flat))

    # Accumulators, all exact.
    hist_n = np.zeros(U_BINS, dtype=np.int64)
    hist_bits = np.zeros(U_BINS, dtype=np.float64)
    hist_n_dis = np.zeros(U_BINS, dtype=np.int64)
    hist_bits_dis = np.zeros(U_BINS, dtype=np.float64)
    # Cost histogram for the disagreeing positions (bits, 0.25-bit bins to 64).
    cost_bins = np.zeros(257, dtype=np.int64)

    total_bits = 0.0
    total_positions = 0
    total_disagree = 0

    corrected_digest = hashlib.sha256()
    cdf_digest = hashlib.sha256()
    started = time.time()

    with torch.inference_mode():
        optimize_sparse_evaluator(sparse)
        previous = torch.zeros(
            (1, runtime.EVAL_H, runtime.EVAL_W), dtype=torch.long, device=device
        )
        for frame in range(runtime.N):
            index = torch.tensor([frame], dtype=torch.long, device=device)
            truth = torch.from_numpy(tokens[frame].astype(np.int64))
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(index, previous)
            if frame:
                previous_cpu = previous[0].to(device="cpu", dtype=torch.uint8).numpy()
                boundary = _boundary_buckets(previous_cpu).reshape(-1)
            else:
                boundary = np.full(runtime.EVAL_H * runtime.EVAL_W, 4, dtype=np.uint8)
            truth_flat = truth.reshape(-1).numpy()
            for group, (device_positions, flat_positions) in enumerate(group_plans):
                selected = sparse.selected_logits(current, context, group)
                base_logits = selected.cpu().numpy()
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                feature = boundary[flat_positions].astype(np.int64) * NUM_CLASSES + predicted
                corrected = base_logits + parts.table.values[feature]
                corrected_digest.update(np.ascontiguousarray(corrected, dtype="<f4").tobytes())
                probability = _probability_table(corrected, runtime.HPAC_LOGIT_PRECISION)
                cdf_digest.update(np.ascontiguousarray(probability, dtype="<f4").tobytes())

                actual = truth_flat[flat_positions]
                prob64 = probability.astype(np.float64)
                p_actual = prob64[np.arange(actual.size), actual]
                arg = prob64.argmax(axis=1)
                p_max = prob64[np.arange(actual.size), arg]
                bits = -np.log2(np.maximum(p_actual, 1e-300))
                disagree = arg != actual

                u = -np.log2(np.maximum(1.0 - p_max, 1e-300))
                idx = np.clip((u / U_STEP).astype(np.int64), 0, U_BINS - 1)
                np.add.at(hist_n, idx, 1)
                np.add.at(hist_bits, idx, bits)
                if disagree.any():
                    np.add.at(hist_n_dis, idx[disagree], 1)
                    np.add.at(hist_bits_dis, idx[disagree], bits[disagree])
                    cb = np.clip((bits[disagree] / 0.25).astype(np.int64), 0, 256)
                    np.add.at(cost_bins, cb, 1)

                total_bits += float(bits.sum())
                total_positions += int(actual.size)
                total_disagree += int(disagree.sum())

                # Teacher forcing: the decoder would have produced exactly `actual`.
                current.reshape(-1)[device_positions] = torch.from_numpy(actual).to(device)
            previous = current
            if frame % 25 == 0:
                el = time.time() - started
                print(
                    f"frame {frame:4d}/{runtime.N}  bits={total_bits:.0f}  "
                    f"disagree={total_disagree}  {el:.0f}s",
                    flush=True,
                )

    control = {
        "corrected_quantized_logit_sha256": corrected_digest.hexdigest(),
        "corrected_cdf_input_sha256": cdf_digest.hexdigest(),
        "expected_corrected_quantized_logit_sha256": receipt["token_decoder"][
            "corrected_quantized_logit_sha256"
        ],
        "expected_corrected_cdf_input_sha256": receipt["token_decoder"][
            "corrected_cdf_input_sha256"
        ],
    }
    control["logit_match"] = (
        control["corrected_quantized_logit_sha256"]
        == control["expected_corrected_quantized_logit_sha256"]
    )
    control["cdf_match"] = (
        control["corrected_cdf_input_sha256"] == control["expected_corrected_cdf_input_sha256"]
    )
    control["instrument_valid"] = bool(control["logit_match"] and control["cdf_match"])

    for name, arr in (
        ("hist_n", hist_n),
        ("hist_bits", hist_bits),
        ("hist_n_disagree", hist_n_dis),
        ("hist_bits_disagree", hist_bits_dis),
        ("cost_bins_disagree", cost_bins),
    ):
        path = RETAINED / f"{name}.npy"
        np.save(path, arr)
        data = path.read_bytes()
        control.setdefault("payloads", {})[name] = {
            "path": str(path),
            "bytes": len(data),
            "sha256": sha256_bytes(data),
        }

    out = {
        "arm": "ddm_rc4",
        "stage": "1_token_rd_replay",
        "archive_sha256": ARCHIVE_SHA,
        "token_stream_bytes": token_stream_bytes,
        "token_stream_bits": token_stream_bytes * 8,
        "retained_decoder_bit_position": receipt["token_decoder"]["decoder_bit_position"],
        "positions": total_positions,
        "ideal_code_bits": total_bits,
        "ideal_vs_shipped_bits_ratio": total_bits / (token_stream_bytes * 8),
        "disagree_positions": total_disagree,
        "top1_error": total_disagree / total_positions,
        "u_step": U_STEP,
        "u_bins": U_BINS,
        "positive_control": control,
        "elapsed_seconds": time.time() - started,
    }
    (STORE / "TOKEN_RD_REPLAY.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "positive_control"}, indent=2))
    print("instrument_valid:", control["instrument_valid"])
    return 0 if control["instrument_valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
