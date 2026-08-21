#!/usr/bin/env python3
"""Measure hash-constrained decode search on real fx5 HPAC token regions.

This is a scorer-free rate experiment, not a score evaluator and not a shipping
candidate.  It imports the exact fx5 HPAC model and adaptive probability law,
walks frame 0 along the retained decoded-token trajectory, and then performs an
exhaustive best-first search over fixed prefixes of real conditional groups.

Every candidate sequence, probability, search order, digest, transmitted
question, and decoded answer is retained under ``--store``.  Each region is a
separate resumable stage.  ``--resume-from`` is required even for a fresh run so
the invocation cannot accidentally grow a non-resumable long sweep later.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
JG2 = REPO / "experiments/ddm_jg2_tail_reencode.py"
DEFAULT_STORE = Path("/Volumes/APDataStore/pact/ddm_dc1_decode_time_compute")
DEFAULT_RUNTIME = Path("/Volumes/APDataStore/pact/ddm_fx5/candidate_runtime_fx5")
DEFAULT_TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_fx5/decode_r1/inflated/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
DEFAULT_LEDGER = Path("/Volumes/APDataStore/pact/ddm_fx5/work/bits_per_frame_e1_19member.npy")
DEFAULT_CONTROL = Path("/Volumes/APDataStore/pact/ddm_fx5/retained/S1_control_600.json")
EXPECTED_ARCHIVE_SHA256 = "4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841"
EXPECTED_TOKEN_SHA256 = (
    "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"  # gitleaks:allow -- public content digest
)
QUESTION_DOMAIN = b"ddm_dc1_hash_region_v1\x00"
NUM_CLASSES = 5


class Dc1Error(RuntimeError):
    """Fail-closed prototype error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + f".partial.{os.getpid()}")
    partial.write_bytes(payload)
    os.replace(partial, path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode())


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + f".partial.{os.getpid()}")
    with partial.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
    os.replace(partial, path)


def load_jg2():
    spec = importlib.util.spec_from_file_location("ddm_dc1_jg2", JG2)
    if spec is None or spec.loader is None:
        raise Dc1Error(f"cannot load {JG2}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prefix_equal(left: np.ndarray, right: np.ndarray, bits: int) -> bool:
    """Whether two uint8 SHA-256 digests share their first ``bits`` bits."""
    if bits == 0:
        return True
    whole, remainder = divmod(bits, 8)
    if whole and not np.array_equal(left[:whole], right[:whole]):
        return False
    if remainder:
        mask = 0xFF & (0xFF << (8 - remainder))
        return int(left[whole]) & mask == int(right[whole]) & mask
    return True


def common_prefix_bits(left: np.ndarray, right: np.ndarray) -> int:
    """Count equal leading bits in two same-length byte vectors."""
    xor = np.bitwise_xor(left, right)
    nonzero = np.flatnonzero(xor)
    if nonzero.size == 0:
        return int(xor.size * 8)
    byte = int(nonzero[0])
    return byte * 8 + (8 - int(int(xor[byte]).bit_length()))


def candidate_digest(group: int, symbols: int, candidate: np.ndarray) -> bytes:
    header = QUESTION_DOMAIN + struct.pack("<HB", group, symbols)
    return hashlib.sha256(header + np.asarray(candidate, dtype=np.uint8).tobytes()).digest()


def pack_question(digest: np.ndarray, bits: int) -> bytes:
    """Concrete byte framing: uint16 bit count plus the left-aligned hash prefix."""
    if bits < 0 or bits > 256:
        raise Dc1Error(f"invalid question width {bits}")
    prefix = bytearray(np.asarray(digest[: (bits + 7) // 8], dtype=np.uint8).tobytes())
    if bits % 8 and prefix:
        prefix[-1] &= 0xFF & (0xFF << (8 - bits % 8))
    return struct.pack("<H", bits) + bytes(prefix)


def pack_bits(bits: list[int]) -> bytes:
    """Pack a list of 0/1 integers MSB-first, padding only the final byte."""
    payload = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        if bit:
            payload[index // 8] |= 1 << (7 - index % 8)
    return bytes(payload)


def unpack_bits(payload: bytes, count: int) -> list[int]:
    """Inverse of :func:`pack_bits` for a caller-provided meaningful bit count."""
    if count < 0 or count > len(payload) * 8:
        raise Dc1Error("packed bit count is outside the payload")
    return [(payload[index // 8] >> (7 - index % 8)) & 1 for index in range(count)]


def integer_bits(value: int, width: int) -> list[int]:
    if width < 0 or value < 0 or value >= (1 << width if width else 1):
        raise Dc1Error("integer does not fit its declared bit width")
    return [(value >> (width - 1 - index)) & 1 for index in range(width)]


def bits_integer(bits: list[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def rank_combination(values: list[int], population: int) -> int:
    """Lexicographic rank of a sorted fixed-cardinality subset."""
    rank = 0
    previous = -1
    cardinality = len(values)
    for index, value in enumerate(values):
        if value <= previous or value >= population:
            raise Dc1Error("combination is not strictly sorted inside the population")
        remaining = cardinality - index - 1
        for candidate in range(previous + 1, value):
            rank += math.comb(population - candidate - 1, remaining)
        previous = value
    return rank


def unrank_combination(rank: int, population: int, cardinality: int) -> list[int]:
    """Inverse of :func:`rank_combination`."""
    total = math.comb(population, cardinality)
    if rank < 0 or rank >= total:
        raise Dc1Error("combination rank is outside its support")
    values: list[int] = []
    previous = -1
    for index in range(cardinality):
        remaining = cardinality - index - 1
        for candidate in range(previous + 1, population):
            count = math.comb(population - candidate - 1, remaining)
            if rank < count:
                values.append(candidate)
                previous = candidate
                break
            rank -= count
        else:  # pragma: no cover - guarded by the support check above.
            raise Dc1Error("failed to unrank a valid combination")
    return values


def decode_question(question: bytes, digests: np.ndarray, order: np.ndarray) -> int:
    """Return the first best-probability candidate satisfying the question."""
    if len(question) < 2:
        raise Dc1Error("question is missing its bit-count header")
    bits = int(struct.unpack_from("<H", question)[0])
    expected = 2 + (bits + 7) // 8
    if len(question) != expected:
        raise Dc1Error(f"question length {len(question)} != framed length {expected}")
    prefix = np.zeros(32, dtype=np.uint8)
    if bits:
        prefix[: (bits + 7) // 8] = np.frombuffer(question[2:], dtype=np.uint8)
    for candidate_index in order:
        index = int(candidate_index)
        if prefix_equal(digests[index], prefix, bits):
            return index
    raise Dc1Error("no candidate satisfies a question derived from the target")


def next_attempt(stage: Path) -> tuple[Path, dict[str, Any] | None]:
    """Reuse a complete stage; otherwise allocate a new retained attempt."""
    if stage.is_dir():
        complete = sorted(stage.glob("attempt_*/receipt.json"), reverse=True)
        for receipt in complete:
            payload = json.loads(receipt.read_text())
            files = payload.get("files", [])
            if payload.get("implementation_sha256") == sha256_file(Path(__file__)) and all(
                Path(str(fact["path"])).is_file()
                and Path(str(fact["path"])).stat().st_size == int(fact["bytes"])
                and sha256_file(Path(str(fact["path"]))) == str(fact["sha256"])
                for fact in files
            ):
                return receipt.parent, payload
        numbers = [int(path.name.split("_")[-1]) for path in stage.glob("attempt_*")]
        number = max(numbers, default=0) + 1
    else:
        number = 1
    attempt = stage / f"attempt_{number:04d}"
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt, None


def extract_rows(
    runtime_root: Path,
    tokens_path: Path,
    groups: list[int],
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, object]]:
    """Walk real frame 0 and return exact adaptive coding rows for selected groups."""
    import torch

    jg2 = load_jg2()
    residual, renderer, renderer_dir = jg2.load_runtime(runtime_root)
    parts = residual.read_residual_archive(runtime_root / "archive.zip")
    device = torch.device("cpu")
    base_hpac = residual.materialize_ihs1(parts.hpac_blob, renderer)
    model = renderer.load_hpac(base_hpac, device)
    masks = renderer.group_masks(device)
    if any(group < 0 or group >= len(masks) for group in groups):
        raise Dc1Error(f"groups must be in 0..{len(masks) - 1}")
    sparse = residual._sparse_class(renderer_dir)(model, renderer.EVAL_H, renderer.EVAL_W)
    from runtime.free_corrector import FreeCorrector
    from runtime.hpac_inference import optimize_sparse_evaluator

    token_shape = (renderer.N, renderer.EVAL_H, renderer.EVAL_W)
    tokens = np.memmap(tokens_path, dtype=np.uint8, mode="r", shape=token_shape)
    target = np.asarray(tokens[0], dtype=np.uint8).reshape(-1)
    group_plans = []
    for mask in masks:
        flat = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1))
        group_plans.append((torch.from_numpy(flat).to(device), flat))

    selected_rows: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    corrector = FreeCorrector(renderer.EVAL_H * renderer.EVAL_W)
    previous = torch.zeros((1, renderer.EVAL_H, renderer.EVAL_W), dtype=torch.long, device=device)
    current = torch.zeros_like(previous)
    context = model.prepare_frame_context(torch.tensor([0], dtype=torch.long), previous)
    boundary = np.full(renderer.EVAL_H * renderer.EVAL_W, 4, dtype=np.uint8)
    corrector.begin_frame(boundary)
    frame_bits = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        optimize_sparse_evaluator(sparse)
        for group, (device_positions, flat_positions) in enumerate(group_plans):
            base_logits = sparse.selected_logits(current, context, group).cpu().numpy()
            predicted = base_logits.argmax(axis=1).astype(np.int64)
            feature = boundary[flat_positions].astype(np.int64) * NUM_CLASSES + predicted
            corrected = base_logits + parts.table.values[feature]
            probability = residual._probability_table(corrected, renderer.HPAC_LOGIT_PRECISION)
            state = corrector.group_state(probability, predicted, flat_positions)
            coding = np.asarray(corrector.coding_row(state), dtype=np.float64)
            symbols = target[flat_positions].astype(np.int64)
            picked = coding[np.arange(coding.shape[0]), symbols]
            frame_bits += float(-np.log2(np.maximum(picked, 1e-300)).sum())
            if group in groups:
                selected_rows[group] = (
                    coding.copy(),
                    symbols.astype(np.uint8),
                    flat_positions.astype(np.int64),
                )
            corrector.observe(state, symbols)
            current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(device)

    reconstructed = current[0].to(device="cpu", dtype=torch.uint8).numpy().reshape(-1)
    if not np.array_equal(reconstructed, target):
        raise Dc1Error("frame-0 HPAC walk diverged from the retained fx5 token field")
    if set(selected_rows) != set(groups):
        raise Dc1Error("one or more requested groups were not materialized")
    return selected_rows, {
        "group_count": len(group_plans),
        "frame": 0,
        "frame_ideal_bits": frame_bits,
        "row_extraction_seconds": time.perf_counter() - started,
    }


def retain_source_rows(
    store: Path,
    rows: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    extraction: dict[str, object],
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, object]]:
    attempt, existing = next_attempt(store / "stages/source_rows")
    if existing is not None:
        loaded: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for group in existing["groups"]:
            loaded[int(group)] = (
                np.load(attempt / f"group_{int(group):03d}_rows.npy", allow_pickle=False),
                np.load(attempt / f"group_{int(group):03d}_target.npy", allow_pickle=False),
                np.load(attempt / f"group_{int(group):03d}_positions.npy", allow_pickle=False),
            )
        return loaded, existing
    files: list[dict[str, object]] = []
    for group, (coding, target, positions) in sorted(rows.items()):
        for suffix, array in (
            ("rows", coding),
            ("target", target),
            ("positions", positions),
        ):
            path = attempt / f"group_{group:03d}_{suffix}.npy"
            atomic_npy(path, array)
            files.append(file_fact(path))
    receipt = {
        "schema": "ddm_dc1_source_rows.v1",
        "implementation_sha256": sha256_file(Path(__file__)),
        "groups": sorted(rows),
        **extraction,
        "files": files,
    }
    atomic_json(attempt / "receipt.json", receipt)
    return rows, receipt


def run_region(
    store: Path,
    group: int,
    symbols: int,
    coding: np.ndarray,
    target: np.ndarray,
    positions: np.ndarray,
    start: int,
    selection_mode: str,
) -> dict[str, object]:
    stage = store / (f"stages/{selection_mode}/group_{group:03d}_start_{start:04d}_symbols_{symbols:02d}")
    attempt, existing = next_attempt(stage)
    if existing is not None:
        return existing

    started = time.perf_counter()
    rows = np.asarray(coding[start : start + symbols], dtype=np.float64)
    answer = np.asarray(target[start : start + symbols], dtype=np.uint8)
    support = NUM_CLASSES**symbols
    powers = (NUM_CLASSES ** np.arange(symbols - 1, -1, -1)).astype(np.int64)
    answer_code = int(np.dot(answer.astype(np.int64), powers))
    map_answer = rows.argmax(axis=1).astype(np.uint8)
    if np.array_equal(answer, map_answer):
        # Optimal early exit: the decoder needs one candidate, not a 5**n table,
        # when the answer is the public model's first best-first result.
        candidates = answer[None, :]
        candidate_codes = np.array([answer_code], dtype=np.int64)
        log_probability = np.array(
            [sum(math.log2(max(rows[i, int(answer[i])], 1e-300)) for i in range(symbols))],
            dtype=np.float64,
        )
        order = np.array([0], dtype=np.int64)
        target_candidate_index = 0
        target_rank = 0
    else:
        candidate_codes = np.arange(support, dtype=np.int64)
        candidates = ((candidate_codes[:, None] // powers[None, :]) % NUM_CLASSES).astype(np.uint8)
        log_probability = np.zeros(support, dtype=np.float64)
        for column in range(symbols):
            probability = rows[column, candidates[:, column]]
            log_probability += np.log2(np.maximum(probability, 1e-300))
        order = np.lexsort((candidate_codes, -log_probability)).astype(np.int64)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(support, dtype=np.int64)
        target_candidate_index = answer_code
        target_rank = int(inverse[target_candidate_index])
    digests = np.empty((len(candidates), 32), dtype=np.uint8)
    for index, candidate in enumerate(candidates):
        digests[index] = np.frombuffer(candidate_digest(group, symbols, candidate), dtype=np.uint8)
    target_digest = digests[target_candidate_index]
    if target_rank == 0:
        hash_bits = 0
        max_earlier_prefix = -1
    else:
        earlier = order[:target_rank]
        prefixes = np.fromiter(
            (common_prefix_bits(digests[int(index)], target_digest) for index in earlier),
            dtype=np.int16,
            count=target_rank,
        )
        max_earlier_prefix = int(prefixes.max())
        hash_bits = max_earlier_prefix + 1

    question = pack_question(target_digest, hash_bits)
    search_started = time.perf_counter()
    decoded_code = decode_question(question, digests, order)
    search_seconds = time.perf_counter() - search_started
    decoded = candidates[decoded_code]
    exact = bool(np.array_equal(decoded, answer))
    if not exact:
        raise Dc1Error("minimal distinguishing hash did not decode the real target")

    arrays = {
        "coding_rows.npy": rows,
        "target.npy": answer,
        "positions.npy": np.asarray(positions[start : start + symbols], dtype=np.int64),
        "candidates.npy": candidates,
        "candidate_codes.npy": candidate_codes,
        "log_probability.npy": log_probability,
        "best_first_order.npy": order,
        "candidate_sha256.npy": digests,
        "decoded.npy": decoded,
    }
    files: list[dict[str, object]] = []
    for name, array in arrays.items():
        path = attempt / name
        atomic_npy(path, array)
        files.append(file_fact(path))
    question_path = attempt / "question.bin"
    atomic_bytes(question_path, question)
    files.append(file_fact(question_path))
    selector = struct.pack("<H", start) if selection_mode == "first-surprise" else b""
    selector_path = attempt / "selector.bin"
    atomic_bytes(selector_path, selector)
    files.append(file_fact(selector_path))

    direct_ideal_bits = float(-log_probability[target_candidate_index])
    selector_lower_bound_bits = (
        0
        if selection_mode in {"fixed-first", "fixed-grid"}
        else math.ceil(math.log2(max(1, len(coding) - symbols + 1)))
    )
    receipt = {
        "schema": "ddm_dc1_hash_region.v1",
        "implementation_sha256": sha256_file(Path(__file__)),
        "group": group,
        "symbols": symbols,
        "selection_mode": selection_mode,
        "selection_start_within_group": start,
        "selection_first_flat_position": int(positions[start]),
        "selection_lower_bound_bits": selector_lower_bound_bits,
        "selector_framed_bytes": len(selector),
        "support_candidates": support,
        "materialized_candidates": len(candidates),
        "target_candidate_code": answer_code,
        "target_candidate_index": target_candidate_index,
        "target_rank_zero_based": target_rank,
        "candidates_examined": target_rank + 1,
        "direct_ideal_bits": direct_ideal_bits,
        "ordinal_binary_lower_bound_bits": (0 if target_rank == 0 else math.floor(math.log2(target_rank)) + 1),
        "minimal_exact_hash_bits": hash_bits,
        "max_earlier_hash_prefix_bits": max_earlier_prefix,
        "framed_question_bytes": len(question),
        "hash_minus_direct_ideal_bits": hash_bits - direct_ideal_bits,
        "framed_minus_direct_ideal_bits": len(question) * 8 - direct_ideal_bits,
        "hash_plus_selector_minus_direct_ideal_bits": (hash_bits + selector_lower_bound_bits - direct_ideal_bits),
        "framed_total_bytes": len(question) + len(selector),
        "exact_roundtrip": exact,
        "search_seconds_after_order": search_seconds,
        "total_stage_seconds": time.perf_counter() - started,
        "files": files,
    }
    atomic_json(attempt / "receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--groups", default="0,38,76,114,152,189")
    parser.add_argument("--max-symbols", type=int, default=8)
    parser.add_argument(
        "--selection-mode",
        choices=("fixed-first", "first-surprise", "fixed-grid"),
        default="fixed-first",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = args.store.resolve()
    resume = args.resume_from.resolve()
    if resume != (store / "manifest.json"):
        raise Dc1Error("--resume-from must name <store>/manifest.json")
    if args.max_symbols < 1 or args.max_symbols > 8:
        raise Dc1Error("--max-symbols must be in 1..8; larger exhaustive supports are unsafe")
    groups = sorted({int(item) for item in args.groups.split(",") if item.strip()})
    archive = args.runtime_root / "archive.zip"
    if sha256_file(archive) != EXPECTED_ARCHIVE_SHA256:
        raise Dc1Error("runtime archive is not the live fx5_e1 object")
    if sha256_file(args.tokens) != EXPECTED_TOKEN_SHA256:
        raise Dc1Error("token checkpoint is not the live fx5 decoded field")
    control = json.loads(args.control.read_text())
    if not control.get("byte_identical"):
        raise Dc1Error("the retained 600-frame real-coder control is not byte-identical")
    ledger = np.load(args.ledger, allow_pickle=False)

    store.mkdir(parents=True, exist_ok=True)
    rows, extraction = extract_rows(args.runtime_root, args.tokens, groups)
    if not math.isclose(float(extraction["frame_ideal_bits"]), float(ledger[0]), rel_tol=0.0, abs_tol=1e-9):
        raise Dc1Error("extracted frame-0 rows disagree with fx5's retained ideal-bit ledger")
    rows, source_receipt = retain_source_rows(store, rows, extraction)

    results: list[dict[str, object]] = []
    for group in groups:
        coding, target, positions = rows[group]
        if args.selection_mode == "fixed-first":
            starts = [0]
            symbol_counts = range(1, args.max_symbols + 1)
        elif args.selection_mode == "first-surprise":
            surprises = np.flatnonzero(coding.argmax(axis=1) != target)
            if surprises.size == 0:
                raise Dc1Error(f"group {group} has no surprise for --selection-mode first-surprise")
            start = int(surprises[0])
            if start + args.max_symbols > len(target):
                start = len(target) - args.max_symbols
            starts = [start]
            symbol_counts = range(1, args.max_symbols + 1)
        else:
            starts = list(range(0, len(target) - args.max_symbols + 1, args.max_symbols))
            symbol_counts = (args.max_symbols,)
        for start in starts:
            for symbols in symbol_counts:
                results.append(
                    run_region(
                        store,
                        group,
                        symbols,
                        coding,
                        target,
                        positions,
                        start,
                        args.selection_mode,
                    )
                )
                atomic_json(
                    resume,
                    {
                        "schema": "ddm_dc1_decode_time_compute_manifest.v1",
                        "completed_regions": len(results),
                        "last_group": group,
                        "last_start": start,
                        "last_symbols": symbols,
                    },
                )

    summary = {
        "schema": "ddm_dc1_decode_time_compute.v1",
        "implementation_sha256": sha256_file(Path(__file__)),
        "axis": "[macOS-CPU advisory / scorer-free real-fx5 token-region measurement]",
        "score_claim": False,
        "archive": file_fact(archive),
        "tokens": file_fact(args.tokens),
        "real_coder_control": file_fact(args.control),
        "ideal_bit_ledger": file_fact(args.ledger),
        "source_rows": source_receipt,
        "groups": groups,
        "max_symbols": args.max_symbols,
        "selection_mode": args.selection_mode,
        "regions": results,
        "aggregate": {
            "region_count": len(results),
            "exact_roundtrips": sum(bool(row["exact_roundtrip"]) for row in results),
            "direct_ideal_bits": sum(float(row["direct_ideal_bits"]) for row in results),
            "minimal_exact_hash_bits": sum(int(row["minimal_exact_hash_bits"]) for row in results),
            "framed_question_bytes": sum(int(row["framed_question_bytes"]) for row in results),
            "selection_lower_bound_bits": sum(int(row["selection_lower_bound_bits"]) for row in results),
            "hash_plus_selector_bits": sum(
                int(row["minimal_exact_hash_bits"]) + int(row["selection_lower_bound_bits"]) for row in results
            ),
            "framed_total_bytes": sum(int(row["framed_total_bytes"]) for row in results),
            "candidates_examined": sum(int(row["candidates_examined"]) for row in results),
            "search_seconds_after_order": sum(float(row["search_seconds_after_order"]) for row in results),
            "total_stage_seconds": sum(float(row["total_stage_seconds"]) for row in results),
        },
    }
    if args.selection_mode == "fixed-grid":
        grids: list[dict[str, object]] = []
        for group in groups:
            group_rows = [row for row in results if int(row["group"]) == group]
            group_rows.sort(key=lambda row: int(row["selection_start_within_group"]))
            non_map = [row for row in group_rows if int(row["target_rank_zero_based"]) > 0]
            width = max((int(row["minimal_exact_hash_bits"]) for row in non_map), default=0)
            bitstream: list[int] = [int(int(row["target_rank_zero_based"]) > 0) for row in group_rows]
            for row in group_rows:
                receipt_path = Path(str(row["files"][0]["path"])).parent
                answer_code = int(row.get("target_candidate_index", row["target_candidate_code"]))
                digests = np.load(receipt_path / "candidate_sha256.npy", allow_pickle=False)
                if int(row["target_rank_zero_based"]) > 0:
                    digest = digests[answer_code]
                    for bit in range(width):
                        bitstream.append((int(digest[bit // 8]) >> (7 - bit % 8)) & 1)
            body = pack_bits(bitstream)
            header = b"DC1G" + struct.pack("<HHH", group, len(group_rows), width)
            packet = header + body
            packet_path = store / f"retained/fixed_grid_group_{group:03d}.bin"
            decoded_path = store / f"retained/fixed_grid_group_{group:03d}_decoded.u8"
            atomic_bytes(packet_path, packet)

            # Parse and consume the RETAINED packet bytes.  Reusing the target digest
            # from construction here would only prove the in-memory plan, not the wire.
            received = packet_path.read_bytes()
            if received[:4] != b"DC1G":
                raise Dc1Error("fixed-grid packet magic changed after persistence")
            wire_group, wire_blocks, wire_width = struct.unpack_from("<HHH", received, 4)
            if (wire_group, wire_blocks, wire_width) != (group, len(group_rows), width):
                raise Dc1Error("fixed-grid packet header did not parse back")
            received_bits = unpack_bits(received[10:], len(bitstream))
            flags = received_bits[:wire_blocks]
            cursor = wire_blocks
            decoded_blocks: list[np.ndarray] = []
            for flag, row in zip(flags, group_rows, strict=True):
                receipt_path = Path(str(row["files"][0]["path"])).parent
                digests = np.load(receipt_path / "candidate_sha256.npy", allow_pickle=False)
                order = np.load(receipt_path / "best_first_order.npy", allow_pickle=False)
                candidates = np.load(receipt_path / "candidates.npy", allow_pickle=False)
                if flag:
                    prefix_bits = received_bits[cursor : cursor + wire_width]
                    cursor += wire_width
                    prefix_bytes = pack_bits(prefix_bits)
                    question = struct.pack("<H", wire_width) + prefix_bytes
                    decoded_code = decode_question(question, digests, order)
                else:
                    decoded_code = int(order[0])
                answer_code = int(row.get("target_candidate_index", row["target_candidate_code"]))
                decoded_blocks.append(candidates[decoded_code])
                if decoded_code != answer_code:
                    raise Dc1Error("persisted fixed-grid packet failed an exact block roundtrip")
            if cursor != len(received_bits):
                raise Dc1Error("fixed-grid packet left meaningful bits unconsumed")
            atomic_bytes(decoded_path, np.concatenate(decoded_blocks).astype(np.uint8).tobytes())

            # Stronger sparse wire: combinatorially rank the non-MAP block set
            # instead of paying one flag per overwhelmingly MAP block.
            non_map_indices = [index for index, row in enumerate(group_rows) if int(row["target_rank_zero_based"]) > 0]
            combination_support = math.comb(len(group_rows), len(non_map_indices))
            combination_width = math.ceil(math.log2(combination_support))
            combination_rank = rank_combination(non_map_indices, len(group_rows))
            sparse_bits = integer_bits(combination_rank, combination_width)
            for index in non_map_indices:
                row = group_rows[index]
                receipt_path = Path(str(row["files"][0]["path"])).parent
                digests = np.load(receipt_path / "candidate_sha256.npy", allow_pickle=False)
                answer_code = int(row.get("target_candidate_index", row["target_candidate_code"]))
                digest = digests[answer_code]
                sparse_bits.extend((int(digest[bit // 8]) >> (7 - bit % 8)) & 1 for bit in range(width))
            sparse_header = b"DC1S" + struct.pack("<HHHH", group, len(group_rows), len(non_map_indices), width)
            sparse_packet = sparse_header + pack_bits(sparse_bits)
            sparse_path = store / f"retained/fixed_grid_sparse_group_{group:03d}.bin"
            sparse_decoded_path = store / (f"retained/fixed_grid_sparse_group_{group:03d}_decoded.u8")
            atomic_bytes(sparse_path, sparse_packet)

            received = sparse_path.read_bytes()
            if received[:4] != b"DC1S":
                raise Dc1Error("sparse fixed-grid packet magic changed after persistence")
            wire_group, wire_blocks, wire_non_map, wire_width = struct.unpack_from("<HHHH", received, 4)
            if (wire_group, wire_blocks, wire_non_map, wire_width) != (
                group,
                len(group_rows),
                len(non_map_indices),
                width,
            ):
                raise Dc1Error("sparse fixed-grid header did not parse back")
            received_sparse_bits = unpack_bits(received[12:], len(sparse_bits))
            cursor = combination_width
            wire_rank = bits_integer(received_sparse_bits[:combination_width])
            wire_indices = unrank_combination(wire_rank, wire_blocks, wire_non_map)
            decoded_blocks = []
            for index, row in enumerate(group_rows):
                receipt_path = Path(str(row["files"][0]["path"])).parent
                digests = np.load(receipt_path / "candidate_sha256.npy", allow_pickle=False)
                order = np.load(receipt_path / "best_first_order.npy", allow_pickle=False)
                candidates = np.load(receipt_path / "candidates.npy", allow_pickle=False)
                if index in wire_indices:
                    prefix = received_sparse_bits[cursor : cursor + wire_width]
                    cursor += wire_width
                    question = struct.pack("<H", wire_width) + pack_bits(prefix)
                    decoded_code = decode_question(question, digests, order)
                else:
                    decoded_code = int(order[0])
                answer_code = int(row.get("target_candidate_index", row["target_candidate_code"]))
                decoded_blocks.append(candidates[decoded_code])
                if decoded_code != answer_code:
                    raise Dc1Error("persisted sparse packet failed an exact block roundtrip")
            if cursor != len(received_sparse_bits):
                raise Dc1Error("sparse fixed-grid packet left meaningful bits unconsumed")
            atomic_bytes(
                sparse_decoded_path,
                np.concatenate(decoded_blocks).astype(np.uint8).tobytes(),
            )
            grids.append(
                {
                    "group": group,
                    "blocks": len(group_rows),
                    "non_map_blocks": len(non_map),
                    "uniform_hash_bits": width,
                    "direct_ideal_bits": sum(float(row["direct_ideal_bits"]) for row in group_rows),
                    "body_bits": len(bitstream),
                    "body_minus_direct_ideal_bits": len(bitstream)
                    - sum(float(row["direct_ideal_bits"]) for row in group_rows),
                    "packet": file_fact(packet_path),
                    "decoded": file_fact(decoded_path),
                    "sparse_combination_bits": combination_width,
                    "sparse_body_bits": len(sparse_bits),
                    "sparse_body_minus_direct_ideal_bits": len(sparse_bits)
                    - sum(float(row["direct_ideal_bits"]) for row in group_rows),
                    "sparse_count_bits_if_not_amortized": math.ceil(math.log2(len(group_rows) + 1)),
                    "sparse_packet": file_fact(sparse_path),
                    "sparse_decoded": file_fact(sparse_decoded_path),
                    "exact_roundtrip": True,
                }
            )
        summary["fixed_grid_packets"] = grids
    atomic_json(store / "retained/result.json", summary)
    atomic_json(
        resume,
        {
            "schema": "ddm_dc1_decode_time_compute_manifest.v1",
            "completed_regions": len(results),
            "expected_regions": len(results),
            "complete": True,
            "result": file_fact(store / "retained/result.json"),
        },
    )
    print(json.dumps(summary["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
