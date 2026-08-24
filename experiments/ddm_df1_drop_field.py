#!/usr/bin/env python3
"""ddm_df1 -- the per-position ``dD/dB`` field for the shipped DX2 token stream.

WHY THIS EXISTS
---------------
Every ``dD/dB`` the campaign owns is a per-rung SCALAR: one aggregate ratio for one
whole edit (W72 46.3x, rf1 478.7x, dg2 687x/792x, tba1-D3 21.62x, ni1 247.69x,
nr1 349x).  Nobody has asked WHERE IN THE OBJECT that cost lives.  Four other
quantities on this same object are violently heavy-tailed (TB2 bit-mass Gini
0.9951593787014772; WJ1 manufactured-error enrichment 257.479902x; BL1 Lane
0.5856% of area carrying 33.5598% of model bits), so the shape of ``dD/dB`` is the
open question that decides whether "shrink the object" is a targeting problem or a
representation problem.

THE GRANULARITY, AND WHY
------------------------
The field is indexed by **token position** -- all 117,964,800 of them, the exact
(600, 384, 512) semantic label lattice that section 4 of the shipped RX1 member
carries at 0.0074392 bits/token.  Three reasons this is the right index and not,
say, "per parameter" or "per tensor":

1. It is what a REAL removal operator can address.  The tail is 62.2% of the
   archive (ddm_tx1 sec.0).  ``cpr1/inflate.py:318`` renders frame ``2p+1`` from
   these tokens and ``upstream/modules.py:108`` (``x = x[:, -1, ...]``) means
   SegNet sees only that frame -- so a token is a near-one-to-one actuator on an
   argmax cell (ddm_jg1 sec.0).  Distortion and rate meet at THIS index.
2. The distortion is measured in the SCORER's D, never weight-MSE.  #1127
   measured render amplification ~38,700x, so a weight-space field would be a
   measurement of nothing.  Here dD is a label change that propagates through the
   shipped ``SemanticTokenRenderer`` and the frozen SegNet.
3. dB at this index is EXACT and already reconciled: BL1/TB2 drove the unmodified
   shipped HPAC fallback + RC64 C decoder and reconciled 910,209.280609 modeled
   bits against 910,216 physical bits with a 6.719391-bit explained residual.

THE OPERATOR
------------
``drop`` -- do not code the token at position ``i``; the receiver substitutes the
model's own argmax.  This is the only removal operator on the tail that needs no
address, because the receiver already computes the coding row before it decodes.

    dB_i = cost_sel_i          (bits the position actually spent)
    dD_i = 0                    if argmax_i == transmitted_i
    dD_i > 0                    otherwise -- the decoded label flips to argmax_i

The identity that makes the cheap half exactly free:  ``cost_sel_i < 1 bit``
<=> ``p_sel_i > 0.5`` <=> ``p_sel_i`` is the unique maximum <=> ``dD_i = 0``.
No probe is needed for that half; it is arithmetic.

WHAT THIS SCRIPT RETAINS
------------------------
One instrumented replay of the shipped decode trajectory, retaining per position
the receiver-visible prediction state that BL1/TB2 did not keep:

  * ``argmax``  u8   -- the final coding-row argmax (NOT the base-logit argmax that
                        TB2's packed RR4 ``base_class`` carries; the corrector mixes)
  * ``pmax``    f32  -- the winning probability, the addressless threshold variable
  * ``psecond`` f32  -- runner-up probability, so the margin field is derivable
  * ``cost``    f64  -- selected cost, re-derived so it can be proven byte-identical
                        to TB2's retained field (the law-identity gate)

Axis: ``[macOS-CPU advisory / scorer-free shipped-receiver instrumentation]``.
``score_claim=false`` -- no scorer runs here, no archive changes, nothing is promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_bl1_per_position_bit_allocation as bl1

APDATA_ROOT = Path("/Volumes/APDataStore/pact")
DEFAULT_STORE = APDATA_ROOT / "ddm_df1_dddb_field" / "measurement_v1"

N = bl1.N
HEIGHT = bl1.HEIGHT
WIDTH = bl1.WIDTH
PLANE = bl1.PLANE
POSITIONS = bl1.POSITIONS
GROUPS = bl1.GROUPS
CLASSES = bl1.CLASSES
STAGE_FRAMES = bl1.STAGE_FRAMES
TOTAL_FREQUENCY = bl1.TOTAL_FREQUENCY

#: TB2's retained selected-cost field.  Byte-identity against it is this arm's proof
#: that it drove the same shipped law and not a re-implementation.
TB2_COST = Path(
    "/Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/measurement_v1/"
    "retained/fields/position_rc64_frequency_cost_bits.f64le.bin"
)
TB2_COST_SHA = "99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86"  # gitleaks:allow -- public content digest

CHECKPOINT_SCHEMA = "ddm_df1_stage_checkpoint.v1"
ESTIMATED_ARTIFACT_BYTES = 6 << 30
RESERVE_BYTES = 8 << 30


class Df1Error(RuntimeError):
    """A fail-closed custody, receiver, resume, or reconciliation error."""


def atomic_json(path: Path, payload: object) -> None:
    bl1.atomic_json(path, payload)


def stage_paths(store: Path, start: int, end: int) -> dict[str, Path]:
    root = store / "retained" / "stages" / f"frames_{start:04d}_{end:04d}"
    return {
        "root": root,
        "argmax": root / "argmax.u8.npy",
        "pmax": root / "pmax.f32.npy",
        "psecond": root / "psecond.f32.npy",
        "cost": root / "cost.f64.npy",
        "state": root / "receiver_state.npz",
        "receipt": root / "RECEIPT.json",
    }


def validate_stage(paths: dict[str, Path], start: int, end: int) -> dict[str, Any]:
    if not paths["receipt"].is_file():
        raise Df1Error(f"missing stage receipt {paths['receipt']}")
    receipt = json.loads(paths["receipt"].read_text())
    if receipt.get("schema") != CHECKPOINT_SCHEMA:
        raise Df1Error(f"stage {start}:{end} carries a foreign schema")
    if int(receipt["frame_start"]) != start or int(receipt["frame_end"]) != end:
        raise Df1Error(f"stage {start}:{end} receipt frame span drifted")
    for key in ("argmax", "pmax", "psecond", "cost", "state"):
        fact = receipt["artifacts"][key]
        path = paths[key]
        if not path.is_file():
            raise Df1Error(f"stage artifact vanished: {path}")
        if path.stat().st_size != int(fact["bytes"]):
            raise Df1Error(f"stage artifact size drifted: {path}")
        if bl1.sha256_file(path) != fact["sha256"]:
            raise Df1Error(f"stage artifact digest drifted: {path}")
    return receipt


def completed_stages(store: Path) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for start in range(0, N, STAGE_FRAMES):
        end = min(start + STAGE_FRAMES, N)
        paths = stage_paths(store, start, end)
        if not paths["receipt"].is_file():
            break
        receipts.append(validate_stage(paths, start, end))
    return receipts


def save_stage(
    paths: dict[str, Path],
    start: int,
    end: int,
    argmax: np.ndarray,
    pmax: np.ndarray,
    psecond: np.ndarray,
    cost: np.ndarray,
    corrector_state: dict[str, np.ndarray],
    decoder_snapshot: np.ndarray,
    previous: np.ndarray,
    elapsed: float,
) -> dict[str, Any]:
    paths["root"].mkdir(parents=True, exist_ok=True)
    bl1.atomic_npy(paths["argmax"], argmax)
    bl1.atomic_npy(paths["pmax"], pmax)
    bl1.atomic_npy(paths["psecond"], psecond)
    bl1.atomic_npy(paths["cost"], cost)
    payload = {f"corrector__{key}": value for key, value in corrector_state.items()}
    payload["schema"] = np.frombuffer(CHECKPOINT_SCHEMA.encode(), dtype=np.uint8)
    payload["frame_end"] = np.asarray([end], dtype=np.int64)
    payload["decoder"] = np.asarray(decoder_snapshot, dtype=np.uint64)
    payload["previous"] = np.asarray(previous, dtype=np.uint8)
    bl1.atomic_npz(paths["state"], payload)
    receipt = {
        "schema": CHECKPOINT_SCHEMA,
        "frame_start": start,
        "frame_end": end,
        "elapsed_seconds": elapsed,
        "cost_bits": float(np.asarray(cost, dtype=np.float64).sum()),
        "argmax_flip_positions": int((argmax != _decoded_slice(start, end)).sum()),
        "artifacts": {
            key: bl1.file_fact(paths[key])
            for key in ("argmax", "pmax", "psecond", "cost", "state")
        },
    }
    atomic_json(paths["receipt"], receipt)
    return receipt


_DECODED: np.memmap | None = None


def _decoded_slice(start: int, end: int) -> np.ndarray:
    """The shipped decoded token field for a frame span, as ``(n, H, W)``."""
    global _DECODED
    if _DECODED is None:
        _DECODED = np.memmap(
            bl1.TO2_TOKENS, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH)
        )
    return np.asarray(_DECODED[start:end])


def coding_prediction(coding: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Receiver-visible prediction state for one group's shipped coding rows.

    Returns ``(argmax, pmax, psecond)``.  The argmax is taken on the SAME float32
    view the shipped ``rc64_costs`` quantizer uses, and with ``np.argmax`` rather
    than ``argsort``, so the winner here is exactly the winner the RC64 frequency
    balance assigns.  That is not pedantry: EXACT float32 ties at the row maximum
    occur at a MEASURED rate of 2.5431315104166665e-06 (10 of 3,932,160 positions in
    frames 0:20), and roughly 70% of those tied rows are flips, so an ``argsort``
    tie-break would manufacture drop-operator flips the real receiver never makes.
    ``np.argmax``'s first-index rule is the receiver's own tie-break because it is
    the rule ``bl1.rc64_costs`` already applies when it balances the winner
    frequency.
    """
    values = np.ascontiguousarray(coding, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != CLASSES:
        raise Df1Error("coding rows do not have the shipped [N,5] shape")
    rows = np.arange(values.shape[0])
    winner = values.argmax(axis=1)
    residual = values.copy()
    residual[rows, winner] = -np.inf
    runner = residual.argmax(axis=1)
    return (
        winner.astype(np.uint8),
        values[rows, winner].astype(np.float32),
        values[rows, runner].astype(np.float32),
    )


def run_decode(store: Path, binding: dict[str, object], library: Path) -> list[dict[str, Any]]:
    import torch

    receipts = completed_stages(store)
    runtime = bl1.load_receiver(binding, library)
    start_frame, previous = _resume(runtime, receipts, store)
    truth = np.memmap(bl1.TO2_TOKENS, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    residual = runtime["residual"]
    renderer = runtime["renderer"]
    parts = runtime["parts"]
    model = runtime["model"]
    sparse = runtime["sparse"]
    plans = runtime["plans"]
    corrector = runtime["corrector"]
    decoder = runtime["decoder"]
    device = runtime["device"]
    jg2 = runtime["jg2"]
    ledger = np.load(bl1.DC1_LEDGER, allow_pickle=False)

    with torch.inference_mode():
        for stage_start in range(start_frame, N, STAGE_FRAMES):
            stage_end = min(stage_start + STAGE_FRAMES, N)
            stage_started = time.perf_counter()
            shape = (stage_end - stage_start, HEIGHT, WIDTH)
            argmax = np.empty(shape, dtype=np.uint8)
            pmax = np.empty(shape, dtype=np.float32)
            psecond = np.empty(shape, dtype=np.float32)
            cost = np.empty(shape, dtype=np.float64)
            model_bits = np.zeros(stage_end - stage_start, dtype=np.float64)
            for frame in range(stage_start, stage_end):
                offset = frame - stage_start
                index = torch.tensor([frame], dtype=torch.long, device=device)
                current = torch.zeros_like(previous)
                frame_context = model.prepare_frame_context(index, previous)
                if frame:
                    previous_cpu = previous[0].to(device="cpu", dtype=torch.uint8).numpy()
                    boundary = residual._boundary_buckets(previous_cpu).reshape(-1)
                else:
                    boundary = np.full(PLANE, 4, dtype=np.uint8)
                corrector.begin_frame(boundary)
                for group, (device_positions, flat_positions) in enumerate(plans):
                    base_logits = sparse.selected_logits(current, frame_context, group).cpu().numpy()
                    predicted = base_logits.argmax(axis=1).astype(np.int64)
                    feature = boundary[flat_positions].astype(np.int64) * CLASSES + predicted
                    corrected = base_logits + parts.table.values[feature]
                    probability = residual._probability_table(corrected, renderer.HPAC_LOGIT_PRECISION)
                    state = corrector.group_state(probability, predicted, flat_positions)
                    coding = np.asarray(corrector.coding_row(state), dtype=np.float64)
                    symbols = decoder.decode(coding).astype(np.int64)
                    expected = np.asarray(truth[frame]).reshape(-1)[flat_positions].astype(np.int64)
                    if not np.array_equal(symbols, expected):
                        raise Df1Error(f"shipped decoder diverged from TO2 at frame={frame} group={group}")
                    freq_bits, _ = bl1.rc64_costs(coding, symbols)
                    win, top, second = coding_prediction(coding)
                    flat = np.asarray(flat_positions)
                    cost[offset].reshape(-1)[flat] = freq_bits
                    argmax[offset].reshape(-1)[flat] = win
                    pmax[offset].reshape(-1)[flat] = top
                    psecond[offset].reshape(-1)[flat] = second
                    model_bits[offset] += float(
                        (-np.log2(coding[np.arange(symbols.size), symbols])).sum()
                    )
                    corrector.observe(state, symbols)
                    current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(device)
                decoded = current[0].to(device="cpu", dtype=torch.uint8).numpy()
                if not np.array_equal(decoded, truth[frame]):
                    raise Df1Error(f"shipped decoded frame {frame} differs from TO2")
                corrector.end_frame(decoded.reshape(-1))
                if not math.isclose(
                    float(model_bits[offset]), float(ledger[frame]), rel_tol=0.0, abs_tol=1e-9
                ):
                    raise Df1Error(f"frame {frame} model cost disagrees with the retained ideal ledger")
                previous = current
            corrector_state = jg2.corrector_state(corrector)
            lost = jg2.uncaptured_divergent_state(
                corrector, runtime["cold_corrector"], set(corrector_state)
            )
            if lost:
                raise Df1Error(f"stage checkpoint would lose adaptive corrector state: {lost[:8]}")
            snapshot = bl1.decoder_state(decoder)
            paths = stage_paths(store, stage_start, stage_end)
            receipt = save_stage(
                paths,
                stage_start,
                stage_end,
                argmax,
                pmax,
                psecond,
                cost,
                corrector_state,
                snapshot,
                previous[0].to(device="cpu", dtype=torch.uint8).numpy(),
                time.perf_counter() - stage_started,
            )
            receipts.append(receipt)
            print(
                json.dumps(
                    {
                        "stage": [stage_start, stage_end],
                        "cost_bits": receipt["cost_bits"],
                        "argmax_flips": receipt["argmax_flip_positions"],
                        "elapsed_s": round(receipt["elapsed_seconds"], 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return receipts


def _resume(runtime: dict[str, Any], receipts: list[dict[str, Any]], store: Path):
    torch = runtime["torch"]
    if not receipts:
        return 0, torch.zeros((1, HEIGHT, WIDTH), dtype=torch.long, device=runtime["device"])
    end = int(receipts[-1]["frame_end"])
    paths = stage_paths(store, end - STAGE_FRAMES, end)
    with np.load(paths["state"], allow_pickle=False) as payload:
        schema = bytes(payload["schema"]).decode()
        if schema != CHECKPOINT_SCHEMA or int(payload["frame_end"][0]) != end:
            raise Df1Error("receiver resume checkpoint schema/frame drifted")
        corrector_payload = {
            key.removeprefix("corrector__"): payload[key].copy()
            for key in payload.files
            if key.startswith("corrector__")
        }
        runtime["jg2"].load_corrector_state(runtime["corrector"], corrector_payload)
        bl1.restore_decoder_state(runtime["decoder"], payload["decoder"])
        previous_np = np.asarray(payload["previous"], dtype=np.uint8).copy()
    previous = (
        torch.from_numpy(previous_np.astype(np.int64))
        .reshape(1, HEIGHT, WIDTH)
        .to(runtime["device"])
    )
    return end, previous


def assemble_fields(store: Path) -> dict[str, dict[str, object]]:
    """Concatenate the staged arrays into flat position-indexed payloads."""
    target = store / "retained" / "fields"
    target.mkdir(parents=True, exist_ok=True)
    layout = {
        "argmax": (target / "position_coding_argmax.u8.bin", np.uint8),
        "pmax": (target / "position_coding_pmax.f32le.bin", np.dtype("<f4")),
        "psecond": (target / "position_coding_psecond.f32le.bin", np.dtype("<f4")),
        "cost": (target / "position_rc64_frequency_cost_bits.f64le.bin", np.dtype("<f8")),
    }
    facts: dict[str, dict[str, object]] = {}
    for key, (path, dtype) in layout.items():
        tmp = path.with_suffix(path.suffix + ".tmp")
        digest = hashlib.sha256()
        written = 0
        with tmp.open("wb") as handle:
            for start in range(0, N, STAGE_FRAMES):
                end = min(start + STAGE_FRAMES, N)
                block = np.load(stage_paths(store, start, end)[key], allow_pickle=False)
                buffer = np.ascontiguousarray(block.reshape(-1), dtype=dtype).tobytes()
                handle.write(buffer)
                digest.update(buffer)
                written += len(buffer)
        os.replace(tmp, path)
        expected = POSITIONS * dtype.itemsize if dtype != np.uint8 else POSITIONS
        if written != expected:
            raise Df1Error(f"assembled {key} field has {written} bytes, expected {expected}")
        facts[key] = {"path": str(path), "bytes": written, "sha256": digest.hexdigest()}
    return facts


def prove_law_identity(facts: dict[str, dict[str, object]]) -> dict[str, object]:
    """The gate: this arm's cost field must be BYTE-IDENTICAL to TB2's retained one.

    Identity proves DF1 drove the same shipped HPAC/corrector/RC64 law rather than a
    look-alike re-implementation, which is the only thing that lets DF1's new
    prediction fields be joined to TB2/WJ1/BL1 aggregates position for position.
    """
    mine = str(facts["cost"]["sha256"])
    theirs = bl1.sha256_file(TB2_COST) if TB2_COST.is_file() else None
    return {
        "df1_cost_sha256": mine,
        "tb2_cost_sha256": theirs,
        "tb2_declared_sha256": TB2_COST_SHA,
        "byte_identical_to_tb2": mine == theirs == TB2_COST_SHA,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    store = args.store.resolve()
    if not store.is_relative_to(APDATA_ROOT.resolve()):
        raise Df1Error(f"output must remain on the APDataStore SSD tier: {APDATA_ROOT}")
    if store.exists() and not args.resume:
        raise Df1Error("output exists; use --resume after verifying its retained receipts")
    store.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(store).free
    if free < ESTIMATED_ARTIFACT_BYTES + RESERVE_BYTES:
        raise Df1Error(f"storage preflight failed: {free} free bytes")
    binding = bl1.source_binding()
    binding["arm"] = "ddm_df1"
    binding["implementation"] = bl1.file_fact(Path(__file__))
    atomic_json(store / "PREFLIGHT.json", binding)
    library = bl1.build_decoder(store)
    receipts = run_decode(store, binding, library)
    facts = assemble_fields(store)
    identity = prove_law_identity(facts)
    if not identity["byte_identical_to_tb2"]:
        raise Df1Error(f"DF1 cost field is not byte-identical to TB2's: {identity}")
    atomic_json(
        store / "DECODE_RESULT.json",
        {
            "schema": "ddm_df1_decode_result.v1",
            "axis": "[macOS-CPU advisory / scorer-free shipped-receiver instrumentation]",
            "score_claim": False,
            "positions": POSITIONS,
            "stages": len(receipts),
            "fields": facts,
            "law_identity": identity,
            "source_binding": binding,
        },
    )
    print(json.dumps({"stages": len(receipts), "law_identity": identity}, indent=2))


if __name__ == "__main__":
    main()
