"""ddm_fs2 stage 0 - exact per-pair rate-distortion structure of the LIVE rc2 token stream.

Re-derives the ddm_rc4 rung-4 replay on the CURRENT pointer body
(`ddm_rc2_composed`, archive df7fd266..., 180,456 B) instead of the retired hv1
body it was originally measured on. Two things change versus rc4:

1.  **Per-pair accumulation.** rc4 accumulated one global histogram over
    `u = -log2(1 - p_max)`. The seventeenth-move candidate is a PER-PAIR
    waterfill (the ddm_jg5 composition law), so the rate leg has to be
    separable per pair. This emits `(N, U_BINS)` histograms, from which the
    exact per-pair rate credit at ANY threshold is a suffix sum.

2.  **Two rate ladders, not one.**
    *  PATH A - "skip decode": the decoder is told to stop decoding wherever
       `p_max >= tau` and substitute the model argmax. Saves the FULL code
       length of every position above tau. This is rc4's ladder. It needs a
       receiver change (a skip-decode branch in `decode_production_tokens`).
    *  PATH B - "argmax substitution": the ENCODER writes the model argmax
       into the token field wherever `p_max >= tau`, and the stream is coded
       with the receiver exactly as it ships today. Saves only the bits
       currently spent at the positions that actually MOVE (the confident
       disagreements), minus the cost of coding the argmax there.

    Both ladders produce the IDENTICAL decoded field, so the seg and pose legs
    are identical; only the rate credit differs. Path B is byte-closeable with
    the proven `ddm_jg2_tail_reencode` machinery and needs no receiver change.

ALWAYS KEEP THE PAYLOAD: the argmax field and the quantised u-index field are
retained in full, so every downstream threshold, drop mask and substituted
token field is a cheap re-derivation and never a re-replay.

THE CORRECTOR IS IN THE LOOP, and it has to be. `decode_production_tokens`
codes against `corrector.coding_row(state)`, which the free corrector adapts
AFTER the two sha256 checkpoints are taken. A first version of this instrument
priced the pre-corrector `_probability_table` output - it reproduced both
digests exactly and still overstated the stream by 2.07% (929,671 ideal bits
against 910,776 shipped), because the corrector is worth 18,895 bits on this
body. On the retired hv1 body that gap was zero, which is why ddm_rc4 never had
to see it. Pricing a token DROP against probabilities the coder does not use
would overstate the credit at exactly the high-confidence positions the drop
targets, so the corrector is stepped here exactly as the decoder steps it -
`begin_frame` / `group_state` / `coding_row` / `observe` / `end_frame` - with
teacher forcing supplying the true symbols.

POSITIVE CONTROLS (fail-closed, all three):
1.  the replay recomputes the shipping decoder's own
    `corrected_quantized_logit_sha256` and `corrected_cdf_input_sha256` and
    refuses unless both match the retained rc2 decode receipt;
2.  the corrector actually selected must be the one the receipt names
    (`NativeFreeCorrector` vs the Python `FreeCorrector` is a different coder);
3.  the summed ideal code length must land within the RC64 quantisation tax of
    the receipt's own `decoder_bit_position`. Control 1 pins the tables;
    control 3 pins the CODER, and only control 3 would have caught the bug
    above.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

GEN = Path("/Volumes/APDataStore/pact/ddm_rc1/candidate_runtime_composed")
STORE = Path("/Volumes/APDataStore/pact/ddm_fs2")
RETAINED = STORE / "retained" / "token_rd"
TOKENS_U8 = Path(
    "/Volumes/APDataStore/pact/ddm_rc2/composed_decode_r2/inflated/"
    ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
TOKENS_RECEIPT = TOKENS_U8.with_suffix(".json")

ARCHIVE_SHA = "df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080"
NUM_CLASSES = 5

# u = -log2(1 - p_max), the natural threshold coordinate near p_max -> 1.
# Identical grid to ddm_rc4 so the two bodies' ladders are directly comparable.
U_MAX = 48.0
U_STEP = 0.125
U_BINS = int(U_MAX / U_STEP) + 2


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _retain(name: str, array: np.ndarray, control: dict) -> None:
    path = RETAINED / f"{name}.npy"
    np.save(path, array)
    data = path.read_bytes()
    control.setdefault("payloads", {})[name] = {
        "path": str(path),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def main() -> int:
    RETAINED.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(GEN))
    sys.path.insert(0, str(GEN / "cpr1"))

    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    from runtime.hpac_inference import optimize_sparse_evaluator
    from runtime.ihs2 import materialize_ihs1
    from runtime.residual_archive import (
        _boundary_buckets,
        _probability_table,
        _rr8_select_corrector,
        _sparse_class,
        read_residual_archive,
    )

    spec = importlib.util.spec_from_file_location("_fs2_renderer", GEN / "cpr1" / "inflate.py")
    runtime = importlib.util.module_from_spec(spec)
    sys.modules["_fs2_renderer"] = runtime
    spec.loader.exec_module(runtime)

    archive = GEN / "archive.zip"
    if sha256_bytes(archive.read_bytes()) != ARCHIVE_SHA:
        raise SystemExit("archive sha mismatch - refusing")
    parts = read_residual_archive(archive)
    token_stream_bytes = len(parts.token_stream)

    receipt = json.loads(TOKENS_RECEIPT.read_text())
    if receipt["binding"]["archive_sha256"] != ARCHIVE_SHA:
        raise SystemExit("retained token checkpoint is not bound to the rc2 archive")
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

    n_pairs = int(runtime.N)
    plane = int(runtime.EVAL_H) * int(runtime.EVAL_W)

    corrector = _rr8_select_corrector(plane)
    corrector_kind = type(corrector).__name__
    expected_kind = receipt["token_decoder"]["free_corrector"]
    if corrector_kind != expected_kind:
        raise SystemExit(
            f"corrector is {corrector_kind}, receipt says {expected_kind}; "
            "a different corrector is a different coder - refusing"
        )

    # Per-pair accumulators - the object rc4 did not build.
    pair_n = np.zeros((n_pairs, U_BINS), dtype=np.int64)
    pair_bits = np.zeros((n_pairs, U_BINS), dtype=np.float64)
    pair_n_dis = np.zeros((n_pairs, U_BINS), dtype=np.int64)
    pair_bits_dis = np.zeros((n_pairs, U_BINS), dtype=np.float64)
    # Full retained fields so any threshold is free downstream.
    argmax_field = np.zeros((n_pairs, plane), dtype=np.uint8)
    u_index_field = np.zeros((n_pairs, plane), dtype=np.uint16)

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
        for frame in range(n_pairs):
            index = torch.tensor([frame], dtype=torch.long, device=device)
            truth = torch.from_numpy(tokens[frame].astype(np.int64))
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(index, previous)
            if frame:
                previous_cpu = previous[0].to(device="cpu", dtype=torch.uint8).numpy()
                boundary = _boundary_buckets(previous_cpu).reshape(-1)
            else:
                boundary = np.full(plane, 4, dtype=np.uint8)
            truth_flat = truth.reshape(-1).numpy()
            corrector.begin_frame(boundary)
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
                # The coder's OWN row, not the pre-corrector table.
                state = corrector.group_state(probability, predicted, flat_positions)
                coding = np.asarray(corrector.coding_row(state))
                corrector.observe(state, actual)

                prob64 = coding.astype(np.float64)
                prob64 /= prob64.sum(axis=1, keepdims=True)
                rows = np.arange(actual.size)
                p_actual = prob64[rows, actual]
                arg = prob64.argmax(axis=1)
                p_max = prob64[rows, arg]
                bits = -np.log2(np.maximum(p_actual, 1e-300))
                disagree = arg != actual

                u = -np.log2(np.maximum(1.0 - p_max, 1e-300))
                idx = np.clip((u / U_STEP).astype(np.int64), 0, U_BINS - 1)
                np.add.at(pair_n[frame], idx, 1)
                np.add.at(pair_bits[frame], idx, bits)
                if disagree.any():
                    np.add.at(pair_n_dis[frame], idx[disagree], 1)
                    np.add.at(pair_bits_dis[frame], idx[disagree], bits[disagree])

                argmax_field[frame, flat_positions] = arg.astype(np.uint8)
                u_index_field[frame, flat_positions] = idx.astype(np.uint16)

                total_bits += float(bits.sum())
                total_positions += int(actual.size)
                total_disagree += int(disagree.sum())

                # Teacher forcing: the decoder would have produced exactly `actual`.
                current.reshape(-1)[device_positions] = torch.from_numpy(actual).to(device)
            corrector.end_frame(
                current[0].to(device="cpu", dtype=torch.uint8).numpy().reshape(-1)
            )
            previous = current
            if frame % 25 == 0:
                elapsed = time.time() - started
                print(
                    f"frame {frame:4d}/{n_pairs}  bits={total_bits:.0f}  "
                    f"disagree={total_disagree}  {elapsed:.0f}s",
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
    # Control 3 - the CODER, not just the tables. The pre-corrector version of this
    # instrument passed controls 1 and 2 and failed this one by 2.07%.
    shipped_bits = float(receipt["token_decoder"]["decoder_bit_position"])
    control["corrector_kind"] = corrector_kind
    control["expected_corrector_kind"] = expected_kind
    control["corrector_match"] = corrector_kind == expected_kind
    control["ideal_code_bits"] = total_bits
    control["decoder_bit_position"] = shipped_bits
    control["ideal_vs_decoded_bits_ratio"] = total_bits / shipped_bits
    control["coder_bits_match"] = bool(abs(total_bits / shipped_bits - 1.0) < 2e-3)
    control["instrument_valid"] = bool(
        control["logit_match"]
        and control["cdf_match"]
        and control["corrector_match"]
        and control["coder_bits_match"]
    )

    for name, arr in (
        ("pair_hist_n", pair_n),
        ("pair_hist_bits", pair_bits),
        ("pair_hist_n_disagree", pair_n_dis),
        ("pair_hist_bits_disagree", pair_bits_dis),
        ("argmax_field", argmax_field),
        ("u_index_field", u_index_field),
    ):
        _retain(name, arr, control)

    out = {
        "arm": "ddm_fs2",
        "stage": "0_rc2_token_rd_replay_per_pair",
        "archive_sha256": ARCHIVE_SHA,
        "body": "ddm_rc2_composed (live pointer)",
        "token_stream_bytes": token_stream_bytes,
        "token_stream_bits": token_stream_bytes * 8,
        "retained_decoder_bit_position": receipt["token_decoder"]["decoder_bit_position"],
        "pairs": n_pairs,
        "positions": total_positions,
        "ideal_code_bits": total_bits,
        "ideal_vs_shipped_bits_ratio": total_bits / (token_stream_bytes * 8),
        "disagree_positions": total_disagree,
        "top1_error": total_disagree / total_positions,
        "u_step": U_STEP,
        "u_bins": U_BINS,
        "positive_control": control,
        "elapsed_seconds": time.time() - started,
        "score_claim": False,
        "promotable": False,
        "axis": "EXACT rate arithmetic on retained decoded field; no scorer forward",
    }
    STORE.mkdir(parents=True, exist_ok=True)
    (STORE / "FS2_TOKEN_RD_REPLAY.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "positive_control"}, indent=2))
    print("instrument_valid:", control["instrument_valid"])
    return 0 if control["instrument_valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
