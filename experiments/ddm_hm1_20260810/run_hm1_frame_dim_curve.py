#!/usr/bin/env python3
"""DDM HM1 real-coder HPAC frame-dimension capacity curve.

This scorer-free instrument measures a receiver-compatible shrink-side model
capacity curve around PR130's deployed D8 HPAC.  Selection uses a seeded,
stratified-random n120 conditional-frame sample.  Every cell persists its
model, exact logits, symbols, Range payload, and decoded symbols.  One
non-control cell is then re-run at n600, packaged, decoded through the real
receiver, and compared frame-by-frame with the retained PR130 raw output.

The n120 rows are selection projections, never score or family authority.  A
trained growth-side curve remains owned by DDM CL1 and is not simulated here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import lzma
import sys
import time
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import constriction
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HP3_PATH = ROOT / "experiments/ddm_hp3_hpac_section_and_zip_frame.py"
DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_hm1_20260810")
REPO_RECEIPT = ROOT / ".omx/research/ddm_hm1_20260810/FINAL_RECEIPT.json"
BASE_RAW = Path(
    "/Volumes/VertigoDataTier/pact/ddm_hp3_20260810/retained/winner_submissions/"
    "requant_frame_embed_step2_hpm300_004436ea59780708_f59bf2e8fe46/inflated/0.raw"
)

SCHEMA = "ddm_hm1_hpac_frame_dim_curve.v1"
AXIS = "[macOS-CPU advisory; scorer-free real serialized bytes]"
SCORE_CLAIM = False
SEED = 20260810
STRATA = 10
SAMPLE_PER_STRATUM = 12
SAMPLE_COUNT = STRATA * SAMPLE_PER_STRATUM
FRAME_COUNT = 600
H = 384
W = 512
K = 5
TOKENS_PER_FRAME = H * W
FULL_TOKEN_COUNT = FRAME_COUNT * TOKENS_PER_FRAME
SAMPLE_TOKEN_COUNT = SAMPLE_COUNT * TOKENS_PER_FRAME
BASE_RAW_BYTES = 3_662_409_600
BASE_RAW_SHA256 = "a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353"
BASE_HPAC_XZ_BYTES = 15_164
BASE_RANGE_BYTES = 116_980
BASE_CAPACITY_JOINT_BYTES = BASE_HPAC_XZ_BYTES + BASE_RANGE_BYTES


class HM1Error(RuntimeError):
    """Raised when an HM1 custody, selection, or receiver invariant fails."""


@dataclass(frozen=True)
class Cell:
    name: str
    frame_dim: int
    kept_dimensions: tuple[int, ...]
    dropped_dimensions: tuple[int, ...]
    raw: bytes
    checkpoint_path: Path
    model_xz_path: Path


def _import_path(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise HM1Error(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _hp3() -> Any:
    return _import_path(HP3_PATH, "ddm_hm1_hp3_helpers")


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _torch_payload(value: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _args_for_dim(frame_dim: int) -> Any:
    class Args:
        channels = 64
        patch = 64
        delta = 2
        weight_bound = 127
        activation_bound = 127
        weight_exponent_min = -6

    Args.frame_dim = frame_dim
    return Args


def stratified_frame_ids(seed: int = SEED) -> np.ndarray:
    if FRAME_COUNT % STRATA:
        raise HM1Error("frame population does not divide into the declared strata")
    rng = np.random.default_rng(seed)
    width = FRAME_COUNT // STRATA
    selected: list[int] = []
    for stratum in range(STRATA):
        start = stratum * width
        draw = rng.choice(np.arange(start, start + width), size=SAMPLE_PER_STRATUM, replace=False)
        selected.extend(sorted(int(value) for value in draw))
    result = np.asarray(selected, dtype=np.int16)
    if result.shape != (SAMPLE_COUNT,) or len(np.unique(result)) != SAMPLE_COUNT:
        raise HM1Error("stratified selection is not a unique n120 sample")
    return result


def _dimension_state(source: dict[str, torch.Tensor], keep: Sequence[int]) -> dict[str, torch.Tensor]:
    index = torch.tensor(tuple(keep), dtype=torch.long)
    result: dict[str, torch.Tensor] = {}
    for name, value in source.items():
        copied = value.detach().clone()
        if name in {"frame_embed.weight", "frame_shift.weight", "frame_scale.weight"}:
            copied = copied.index_select(1, index)
        result[name] = copied
    return result


def _recompute_affected_depths(model: Any, packer: Any) -> None:
    """Tighten only rows whose values changed when conditioning columns were removed."""

    modules = dict(model.named_modules())
    with torch.no_grad():
        for name in ("frame_shift", "frame_scale"):
            module = modules[name]
            rows = packer.module_weight_rows(module, module.codes()[0])
            depths: list[int] = []
            for row in rows:
                low = int(row.min(initial=0))
                high = int(row.max(initial=0))
                selected = 0
                if low != 0 or high != 0:
                    for bits in range(1, 9):
                        bound = getattr(module, "weight_bound", 127)
                        if low >= max(-(1 << (bits - 1)), -bound) and high <= min((1 << (bits - 1)) - 1, bound):
                            selected = bits
                            break
                if selected == 0 and (low != 0 or high != 0):
                    raise HM1Error(f"{name} row lies outside the HPAC 8-bit grammar")
                depths.append(selected)
            module.bit_depth.copy_(torch.tensor(depths, dtype=module.bit_depth.dtype))


def build_cell(
    *,
    hp: Any,
    packer: Any,
    control_state: dict[str, torch.Tensor],
    keep: Sequence[int],
    output: Path,
) -> Cell:
    kept = tuple(sorted(int(value) for value in keep))
    if not kept or len(set(kept)) != len(kept) or any(value < 0 or value >= 8 for value in kept):
        raise HM1Error(f"invalid frame-dimension subset: {kept}")
    dropped = tuple(value for value in range(8) if value not in kept)
    name = f"frame_dim{len(kept)}_drop" + ("none" if not dropped else "_".join(map(str, dropped)))
    root = output / "retained/candidates" / name
    model = packer.model_from_args(_args_for_dim(len(kept)), True)
    model.load_state_dict(_dimension_state(control_state, kept), strict=True)
    for module in packer.compressible_modules(model):
        module.self_compress_deployed = True
    if len(kept) < 8:
        _recompute_affected_depths(model, packer)
    model.eval()
    raw = packer.serialize_self_compressed(model)
    if len(kept) == 8 and _sha256_bytes(raw) != hp.HPAC_RAW_SHA256:
        raise HM1Error("D8 control is not the exact shipped HPAC model")
    raw_record = hp.retain_payload(root / "hpac.raw", raw)
    model_xz = lzma.compress(raw, format=lzma.FORMAT_XZ, filters=hp.LZMA_FILTERS)
    model_xz_path = root / "hpac.xz"
    model_xz_record = hp.retain_payload(model_xz_path, model_xz)
    checkpoint_path = root / "checkpoint.pt"
    checkpoint_record = hp.retain_payload(
        checkpoint_path,
        _torch_payload(
            {
                "state_dict": model.state_dict(),
                "config": {
                    "channels": 64,
                    "patch": 64,
                    "delta": 2,
                    "frame_dim": len(kept),
                    "kept_dimensions": list(kept),
                    "source_checkpoint_sha256": hp.CHECKPOINT_SHA256,
                },
                "score_claim": False,
                "axis": AXIS,
            }
        ),
    )
    restored = packer.model_from_args(_args_for_dim(len(kept)), False).eval()
    packer.deserialize_self_compressed(restored, raw)
    with torch.no_grad():
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        current = torch.randint(0, K, (2, H, W), generator=generator)
        previous = torch.randint(0, K, (2, H, W), generator=generator)
        indices = torch.tensor([0, 599])
        expected = model(current, indices, previous)
        actual = restored(current, indices, previous)
    max_logit_diff = float((expected - actual).abs().max())
    if max_logit_diff != 0.0:
        raise HM1Error(f"{name} packed model changed logits by {max_logit_diff}")
    hp.atomic_json(
        root / "model_receipt.json",
        {
            "schema": "ddm_hm1_model_cell.v1",
            "name": name,
            "frame_dim": len(kept),
            "kept_dimensions": list(kept),
            "dropped_dimensions": list(dropped),
            "hpac_raw": raw_record,
            "hpac_xz": model_xz_record,
            "checkpoint": checkpoint_record,
            "parse_back_exact": True,
            "max_logit_diff": max_logit_diff,
            "axis": AXIS,
            "score_claim": False,
        },
    )
    return Cell(name, len(kept), kept, dropped, raw, checkpoint_path, model_xz_path)


def _restore_cell_model(cell: Cell, packer: Any) -> Any:
    model = packer.model_from_args(_args_for_dim(cell.frame_dim), False).eval()
    packer.deserialize_self_compressed(model, cell.raw)
    return model


def _verify_npy(path: Path, record: dict[str, Any], hp: Any) -> None:
    if not path.is_file() or path.stat().st_size != record["bytes"] or hp.sha256_file(path) != record["sha256"]:
        raise HM1Error(f"retained NumPy payload failed custody: {path}")


@torch.no_grad()
def materialize_frames(
    *,
    hp: Any,
    packer: Any,
    inflate: Any,
    cell: Cell,
    cache: torch.Tensor,
    frame_ids: Sequence[int],
    root: Path,
    stage_frames: int,
    contiguous: bool,
) -> dict[str, Any]:
    manifest_path = root / "code_manifest.json"
    prior_rows: list[dict[str, Any]] = []
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("hpac_sha256") != _sha256_bytes(cell.raw) or prior.get("frame_ids") != list(frame_ids):
            raise HM1Error(f"resume identity changed for {cell.name}")
        prior_rows = list(prior.get("chunks", []))
        if prior.get("complete") is True:
            for row in prior_rows:
                _verify_npy(Path(row["symbols"]["path"]), row["symbols"], hp)
                _verify_npy(Path(row["codes"]["path"]), row["codes"], hp)
            return prior

    model = _restore_cell_model(cell, packer)
    masks = inflate.group_masks(torch.device("cpu"))
    sparse = inflate.SparseIntegerHPAC(model, H, W)
    rows: list[dict[str, Any]] = []
    ids = [int(value) for value in frame_ids]
    for start_index in range(0, len(ids), stage_frames):
        end_index = min(start_index + stage_frames, len(ids))
        stage_ids = ids[start_index:end_index]
        existing = next((row for row in prior_rows if row["start_index"] == start_index), None)
        if existing is not None:
            symbols_path = Path(existing["symbols"]["path"])
            codes_path = Path(existing["codes"]["path"])
            try:
                _verify_npy(symbols_path, existing["symbols"], hp)
                _verify_npy(codes_path, existing["codes"], hp)
                rows.append(existing)
                continue
            except HM1Error:
                pass
        started = time.perf_counter()
        symbol_parts: list[np.ndarray] = []
        code_parts: list[np.ndarray] = []
        for frame in stage_ids:
            previous = torch.zeros((1, H, W), dtype=torch.long) if frame == 0 else cache[frame - 1 : frame].long()
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(torch.tensor([frame]), previous)
            for group, mask in enumerate(masks):
                selected = sparse.selected_logits(current, context, group)
                codes = selected.mul(8).round().clamp(-32768, 32767).to(torch.int16)
                symbols = cache[frame][mask].numpy().astype(np.uint8)
                code_parts.append(codes.numpy())
                symbol_parts.append(symbols)
                current[0, mask] = torch.from_numpy(symbols.astype(np.int64))
        symbols_array = np.concatenate(symbol_parts)
        codes_array = np.concatenate(code_parts)
        expected_tokens = len(stage_ids) * TOKENS_PER_FRAME
        if symbols_array.shape != (expected_tokens,) or codes_array.shape != (expected_tokens, K):
            raise HM1Error(f"materialized geometry changed for {cell.name}")
        suffix = f"{start_index:03d}_{end_index:03d}"
        symbols_path = root / f"symbols_{suffix}.npy"
        codes_path = root / f"codes_{suffix}.npy"
        row = {
            "start_index": start_index,
            "end_index": end_index,
            "frame_ids": stage_ids,
            "symbols": hp.save_npy_atomic(symbols_path, symbols_array),
            "codes": hp.save_npy_atomic(codes_path, codes_array),
            "materialize_wall_s": time.perf_counter() - started,
        }
        if contiguous:
            row["start_frame"] = stage_ids[0]
            row["end_frame"] = stage_ids[-1] + 1
        rows.append(row)
        hp.replace_json(
            manifest_path,
            {
                "schema": "ddm_hp3_code_chunks.v1" if contiguous else "ddm_hm1_sample_code_chunks.v1",
                "complete": end_index == len(ids),
                "candidate": cell.name,
                "hpac_sha256": _sha256_bytes(cell.raw),
                "frame_ids": ids,
                "frames": end_index,
                "tokens": end_index * TOKENS_PER_FRAME,
                "chunks": rows,
                "selection_mode": "full_contiguous" if contiguous else "seeded_stratified_random_n120",
            },
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("tokens") != len(ids) * TOKENS_PER_FRAME:
        raise HM1Error(f"materialization incomplete for {cell.name}")
    return manifest


def _array_blocks(array: np.ndarray, block: int = 65_536) -> Iterable[tuple[int, int]]:
    for start in range(0, len(array), block):
        yield start, min(start + block, len(array))


def encode_sample(*, hp: Any, cell: Cell, manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    token_path = root / "tokens.range"
    decoded_path = root / "tokens.decoded.u8"
    family = constriction.stream.model.Categorical(perfect=False)
    ideal_bits = 0.0
    source_digest = hashlib.sha256()
    if not token_path.exists():
        encoder = constriction.stream.queue.RangeEncoder()
        for row in manifest["chunks"]:
            symbols = np.load(row["symbols"]["path"], mmap_mode="r", allow_pickle=False)
            codes = np.load(row["codes"]["path"], mmap_mode="r", allow_pickle=False)
            source_digest.update(np.asarray(symbols).tobytes(order="C"))
            for start, end in _array_blocks(symbols):
                tables = hp.probability_tables(codes[start:end])
                target = np.asarray(symbols[start:end], dtype=np.int32)
                ideal_bits += float(-np.log2(tables[np.arange(len(target)), target].astype(np.float64)).sum())
                encoder.encode(target, family, tables)
        hp.retain_payload(token_path, encoder.get_compressed().tobytes())
    else:
        for row in manifest["chunks"]:
            symbols = np.load(row["symbols"]["path"], mmap_mode="r", allow_pickle=False)
            codes = np.load(row["codes"]["path"], mmap_mode="r", allow_pickle=False)
            source_digest.update(np.asarray(symbols).tobytes(order="C"))
            for start, end in _array_blocks(symbols):
                tables = hp.probability_tables(codes[start:end])
                target = np.asarray(symbols[start:end], dtype=np.int32)
                ideal_bits += float(-np.log2(tables[np.arange(len(target)), target].astype(np.float64)).sum())

    decoder = constriction.stream.queue.RangeDecoder(np.frombuffer(token_path.read_bytes(), dtype=np.uint32))
    decoded_parts: list[np.ndarray] = []
    for row in manifest["chunks"]:
        symbols = np.load(row["symbols"]["path"], mmap_mode="r", allow_pickle=False)
        codes = np.load(row["codes"]["path"], mmap_mode="r", allow_pickle=False)
        for start, end in _array_blocks(symbols):
            decoded = decoder.decode(family, hp.probability_tables(codes[start:end])).astype(np.uint8)
            if not np.array_equal(decoded, symbols[start:end]):
                raise HM1Error(f"sample Range decode changed symbols for {cell.name}")
            decoded_parts.append(decoded)
    decoded_payload = np.concatenate(decoded_parts).tobytes(order="C")
    decoded_record = hp.retain_payload(decoded_path, decoded_payload)
    if decoded_record["bytes"] != SAMPLE_TOKEN_COUNT or decoded_record["sha256"] != source_digest.hexdigest():
        raise HM1Error(f"sample decoded payload identity failed for {cell.name}")
    multiplier = FRAME_COUNT / SAMPLE_COUNT
    model_bytes = cell.model_xz_path.stat().st_size
    return {
        "candidate": cell.name,
        "frame_dim": cell.frame_dim,
        "kept_dimensions": list(cell.kept_dimensions),
        "dropped_dimensions": list(cell.dropped_dimensions),
        "selection_mode": "seeded_stratified_random_n120; 10 strata x 12 frames",
        "sample_frames": SAMPLE_COUNT,
        "sample_tokens": SAMPLE_TOKEN_COUNT,
        "sample_fraction": SAMPLE_COUNT / FRAME_COUNT,
        "hpac_xz": hp.file_record(cell.model_xz_path),
        "sample_range": hp.file_record(token_path),
        "sample_decoded": decoded_record,
        "sample_decode_exact": True,
        "sample_ideal_bits": ideal_bits,
        "projected_n600_ideal_token_bytes": int(np.ceil(ideal_bits * multiplier / 8.0)),
        "projected_n600_range_token_bytes": round(token_path.stat().st_size * multiplier),
        "projected_n600_ideal_joint_bytes": model_bytes + int(np.ceil(ideal_bits * multiplier / 8.0)),
        "projected_n600_range_joint_bytes": model_bytes + round(token_path.stat().st_size * multiplier),
        "projection_warning": (
            "selection-only Horvitz-Thompson-style 5x token projection; full model counted once; "
            "Range finite-state/framing effects are remeasured only for the selected n600 cell"
        ),
        "axis": AXIS,
        "score_claim": False,
    }


def _stage_runtime(*, hp: Any, output: Path, archive: Path, frame_dim: int) -> dict[str, Any]:
    stage = output / "retained/winner_submission"
    stage.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for name in ("receiver.py", "hpac_integer.py", "hpac_integer_sparse.py", "integer_model_io.py", "carrier_codec.py"):
        source = hp.FX1_RUNTIME / name
        destination = stage / name
        hp.atomic_bytes(destination, source.read_bytes())
        records.append({"source": str(source), "staged": hp.file_record(destination)})
    base_source = hp.FX1_RUNTIME / "inflate.py"
    base_payload = base_source.read_text(encoding="utf-8")
    old = "HPAC_FILM_DIM = 8"
    if base_payload.count(old) != 1:
        raise HM1Error("FX1 runtime no longer has one frame-dimension constant")
    base_payload = base_payload.replace(old, f"HPAC_FILM_DIM = {frame_dim}")
    hp.atomic_bytes(stage / "inflate_base.py", base_payload.encode())
    records.append(
        {
            "source": str(base_source),
            "adaptation": f"frame_dim={frame_dim}",
            "staged": hp.file_record(stage / "inflate_base.py"),
        }
    )
    for name in ("inflate_hp3.py", "inflate.sh", "hp3_codec.py"):
        source = hp.HP3_RUNTIME / name
        destination = stage / name
        hp.atomic_bytes(destination, source.read_bytes())
        records.append({"source": str(source), "staged": hp.file_record(destination)})
    hp.atomic_bytes(stage / "archive.zip", archive.read_bytes())
    archive_dir = stage / "archive"
    archive_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        hp.retain_payload(archive_dir / "p", handle.read("p"))
    return {"path": str(stage), "files": records, "archive": hp.file_record(stage / "archive.zip")}


def frame_parity_screen(*, hp: Any, candidate_raw: Path, base_raw: Path, output: Path) -> dict[str, Any]:
    if hp.file_record(base_raw) != {"path": str(base_raw), "bytes": BASE_RAW_BYTES, "sha256": BASE_RAW_SHA256}:
        raise HM1Error("retained base raw pin failed")
    candidate_record = hp.file_record(candidate_raw)
    if candidate_record["bytes"] != BASE_RAW_BYTES:
        raise HM1Error("candidate raw geometry differs from the base")
    frame_bytes = BASE_RAW_BYTES // (FRAME_COUNT * 2)
    if frame_bytes * FRAME_COUNT * 2 != BASE_RAW_BYTES:
        raise HM1Error("raw video does not divide into 1200 frames")
    accumulators = {
        "even_pose_carrier": {"frames": 0, "changed_bytes": 0, "abs_sum": 0, "max_abs": 0},
        "odd_semantic": {"frames": 0, "changed_bytes": 0, "abs_sum": 0, "max_abs": 0},
    }
    with base_raw.open("rb") as base_handle, candidate_raw.open("rb") as candidate_handle:
        for frame in range(FRAME_COUNT * 2):
            base = np.frombuffer(base_handle.read(frame_bytes), dtype=np.uint8)
            candidate = np.frombuffer(candidate_handle.read(frame_bytes), dtype=np.uint8)
            if len(base) != frame_bytes or len(candidate) != frame_bytes:
                raise HM1Error("raw frame read truncated")
            delta = np.abs(base.astype(np.int16) - candidate.astype(np.int16))
            key = "even_pose_carrier" if frame % 2 == 0 else "odd_semantic"
            row = accumulators[key]
            row["frames"] += 1
            row["changed_bytes"] += int(np.count_nonzero(delta))
            row["abs_sum"] += int(delta.sum(dtype=np.int64))
            row["max_abs"] = max(row["max_abs"], int(delta.max(initial=0)))
    for row in accumulators.values():
        denominator = row["frames"] * frame_bytes
        row["denominator_bytes"] = denominator
        row["changed_fraction"] = row["changed_bytes"] / denominator
        row["mean_abs_delta"] = row.pop("abs_sum") / denominator
    result = {
        "schema": "ddm_hm1_frame_parity_screen.v1",
        "base_raw": hp.file_record(base_raw),
        "candidate_raw": candidate_record,
        "frame_bytes": frame_bytes,
        "rows": accumulators,
        "byte_identical": candidate_record["sha256"] == BASE_RAW_SHA256,
        "axis": AXIS,
        "score_claim": False,
    }
    hp.atomic_json(output / "FRAME_PARITY_SCREEN.json", result)
    return result


def _pin_source_inputs(hp: Any) -> dict[str, Any]:
    pins = hp.pin_inputs()
    expected_base_raw = {"path": str(BASE_RAW), "bytes": BASE_RAW_BYTES, "sha256": BASE_RAW_SHA256}
    observed_base_raw = hp.file_record(BASE_RAW)
    if observed_base_raw != expected_base_raw:
        raise HM1Error("base raw source pin failed")
    pins["base_raw"] = observed_base_raw
    return pins


def _initialize_state(*, hp: Any, output: Path, state_path: Path, pins: dict[str, Any], sample_ids: np.ndarray) -> None:
    expected = output / "run_state.json"
    if state_path.resolve() != expected.resolve():
        raise HM1Error(f"--resume-from must be {expected}")
    identity = {
        "schema": SCHEMA,
        "instrument_sha256": hp.sha256_file(Path(__file__)),
        "base_archive_sha256": hp.BASE_SHA256,
        "source_checkpoint_sha256": hp.CHECKPOINT_SHA256,
        "sample_seed": SEED,
        "sample_frame_ids": sample_ids.astype(int).tolist(),
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if any(state.get(key) != value for key, value in identity.items()):
            raise HM1Error("resume identity changed")
        return
    hp.atomic_json(
        state_path,
        {
            **identity,
            "complete": False,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "argv": sys.argv,
            "pins": pins,
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    hp = _hp3()
    preflight = hp.storage_preflight(args.output, args.required_free_bytes)
    pins = _pin_source_inputs(hp)
    sample_ids = stratified_frame_ids()
    hp.save_npy_atomic(args.output / "retained/selection/frame_ids.npy", sample_ids)
    hp.atomic_json(
        args.output / "retained/selection/selection.json",
        {
            "schema": "ddm_hm1_stratified_selection.v1",
            "seed": SEED,
            "population_frames": FRAME_COUNT,
            "strata": STRATA,
            "stratum_width": FRAME_COUNT // STRATA,
            "sample_per_stratum": SAMPLE_PER_STRATUM,
            "selected_frames": sample_ids.astype(int).tolist(),
            "selection_mode": "seeded stratified random; never a prefix",
        },
    )
    _initialize_state(hp=hp, output=args.output, state_path=args.resume_from, pins=pins, sample_ids=sample_ids)
    packer, inflate = hp.configure_sources()
    control = hp.model_from_checkpoint(packer)
    control_state = {name: value.detach().clone() for name, value in control.state_dict().items()}
    cache = torch.load(hp.CACHE, map_location="cpu", weights_only=False)["seg"][:FRAME_COUNT].to(torch.uint8)
    if tuple(cache.shape) != (FRAME_COUNT, H, W):
        raise HM1Error("canonical cache geometry changed")

    cells: dict[str, Cell] = {}
    rows: list[dict[str, Any]] = []

    def measure_subset(keep: Sequence[int]) -> dict[str, Any]:
        cell = build_cell(hp=hp, packer=packer, control_state=control_state, keep=keep, output=args.output)
        cells[cell.name] = cell
        sample_root = args.output / "retained/selection" / cell.name
        manifest = materialize_frames(
            hp=hp,
            packer=packer,
            inflate=inflate,
            cell=cell,
            cache=cache,
            frame_ids=sample_ids.astype(int).tolist(),
            root=sample_root,
            stage_frames=12,
            contiguous=False,
        )
        row = encode_sample(hp=hp, cell=cell, manifest=manifest, root=sample_root)
        hp.replace_json(sample_root / "result.json", row)
        rows.append(row)
        hp.replace_json(
            args.output / "SELECTION_CURVE.partial.json",
            {
                "schema": "ddm_hm1_selection_curve.partial.v1",
                "complete": False,
                "rows": rows,
                "selection": hp.file_record(args.output / "retained/selection/selection.json"),
            },
        )
        return row

    control_row = measure_subset(tuple(range(8)))
    d7_rows = [measure_subset(tuple(value for value in range(8) if value != dropped)) for dropped in range(8)]
    best_d7 = min(d7_rows, key=lambda row: row["projected_n600_range_joint_bytes"])
    best_d7_keep = tuple(int(value) for value in best_d7["kept_dimensions"])
    d6_rows = [measure_subset(tuple(value for value in best_d7_keep if value != dropped)) for dropped in best_d7_keep]

    best_projection = min(rows, key=lambda row: row["projected_n600_range_joint_bytes"])
    noncontrol = [row for row in rows if row["frame_dim"] < 8]
    selected = min(noncontrol, key=lambda row: row["projected_n600_range_joint_bytes"])
    selection_disposition = (
        "PROJECTED_WINNER_NONCONTROL"
        if best_projection["candidate"] == selected["candidate"]
        and selected["projected_n600_range_joint_bytes"] < control_row["projected_n600_range_joint_bytes"]
        else "BEST_NONCONTROL_FOR_N600_NEGATIVE_CLOSURE"
    )
    selected_cell = cells[selected["candidate"]]

    full_code_manifest_path = args.output / "retained/codes" / selected_cell.name / "code_manifest.json"
    full_token_payload_path = args.output / "retained/candidates" / selected_cell.name / "tokens.hpm1"
    full_resume_ready = False
    if full_code_manifest_path.exists() and full_token_payload_path.exists():
        retained_manifest = json.loads(full_code_manifest_path.read_text(encoding="utf-8"))
        full_resume_ready = bool(
            retained_manifest.get("complete") is True
            and retained_manifest.get("frames") == FRAME_COUNT
            and retained_manifest.get("tokens") == FULL_TOKEN_COUNT
        )
    hp.storage_preflight(args.output, (4 if full_resume_ready else 7) << 30)
    full_manifest = materialize_frames(
        hp=hp,
        packer=packer,
        inflate=inflate,
        cell=selected_cell,
        cache=cache,
        frame_ids=list(range(FRAME_COUNT)),
        root=args.output / "retained/codes" / selected_cell.name,
        stage_frames=24,
        contiguous=True,
    )
    hp.retain_payload(args.output / "retained/candidates" / selected_cell.name / "hpac.raw", selected_cell.raw)
    full_tokens = hp.encode_monolithic_checkpoint(args.output, selected_cell.name, full_manifest)
    _, semantic_pose, _, _ = hp.split_base()
    candidate_descriptor = hp.CandidateModel(
        selected_cell.name,
        selected_cell.raw,
        False,
        len(selected_cell.dropped_dimensions),
        f"drop deployed frame-conditioning dimensions {selected_cell.dropped_dimensions}; HPM300 receiver checkpoint",
        300,
        selected_cell.name,
    )
    token_payload = Path(full_tokens["payload"]["path"]).read_bytes()
    archive = hp.build_archive_candidate(args.output, candidate_descriptor, semantic_pose, token_payload)
    runtime = _stage_runtime(
        hp=hp,
        output=args.output,
        archive=Path(archive["archive"]["path"]),
        frame_dim=selected_cell.frame_dim,
    )
    receiver_state_path = (
        args.output / "retained/winner_submission/inflated/0.raw.hp3_state/render_state.json"
    )
    receiver_raw_path = args.output / "retained/winner_submission/inflated/0.raw"
    receiver_resume_ready = False
    if receiver_state_path.exists() and receiver_raw_path.exists():
        receiver_state = json.loads(receiver_state_path.read_text(encoding="utf-8"))
        receiver_resume_ready = bool(
            receiver_state.get("complete") is True
            and receiver_raw_path.stat().st_size == BASE_RAW_BYTES
        )
    receiver_preflight = hp.storage_preflight(args.output, (4 if receiver_resume_ready else 5) << 30)
    inflate_result = hp.run_inflate(args.output, runtime)
    parity = frame_parity_screen(
        hp=hp,
        candidate_raw=Path(inflate_result["inflated_raw"]["path"]),
        base_raw=BASE_RAW,
        output=args.output,
    )

    model_xz_bytes = selected_cell.model_xz_path.stat().st_size
    raw_range_bytes = int(full_tokens["range_payload"]["bytes"])
    full_capacity_joint = model_xz_bytes + raw_range_bytes
    result = {
        "schema": SCHEMA,
        "complete": True,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "authority_boundary": "no scorer slot claimed; no upstream/evaluate.py run",
        "pins": pins,
        "storage_preflight": preflight,
        "selection": {
            "seed": SEED,
            "mode": "seeded stratified random n120; 10 strata x 12 frames; never a prefix",
            "frame_ids": sample_ids.astype(int).tolist(),
            "projection_multiplier": FRAME_COUNT / SAMPLE_COUNT,
            "rows": rows,
            "best_d7": best_d7["candidate"],
            "d6_candidates": [row["candidate"] for row in d6_rows],
            "best_projection": best_projection["candidate"],
            "selected_for_n600": selected["candidate"],
            "selection_disposition": selection_disposition,
        },
        "full_n600": {
            "candidate": selected_cell.name,
            "frame_dim": selected_cell.frame_dim,
            "kept_dimensions": list(selected_cell.kept_dimensions),
            "dropped_dimensions": list(selected_cell.dropped_dimensions),
            "hpac_xz": hp.file_record(selected_cell.model_xz_path),
            "raw_range": full_tokens["range_payload"],
            "resumable_token_payload": full_tokens["payload"],
            "model_plus_raw_range_bytes": full_capacity_joint,
            "base_model_plus_raw_range_bytes": BASE_CAPACITY_JOINT_BYTES,
            "delta_model_plus_raw_range_bytes_vs_base": full_capacity_joint - BASE_CAPACITY_JOINT_BYTES,
            "archive": archive,
            "receiver_storage_preflight": receiver_preflight,
            "receiver": inflate_result,
            "frame_parity": parity,
            "decoded_token_sha256": hp.RAW_TOKEN_SHA256,
            "decoded_token_count": FULL_TOKEN_COUNT,
        },
        "free_side_audit": {
            "derived_generic_receiver_objects_already_free": [
                "patch-group masks and sparse gather plans",
                "coordinate grids and causal scan order",
                "integer arithmetic and Range decoder algorithm",
            ],
            "counted_video_derived_objects": [
                "all learned HPAC weights, biases, exponents, and per-row depths",
                "600-by-D frame embeddings",
                "Range-coded semantic tokens",
            ],
            "derive_instead_of_store_finding": "none proven on the exact shipped D8 model",
            "bit_depth_boundary": (
                "per-row depths are needed to parse the variable-bit weight stream before weights exist; "
                "they are not decoder-derivable under IHS1 without a new self-delimiting representation"
            ),
        },
        "trained_growth_side": {
            "status": "QUEUED-WITH-EXISTING-FIRE-ORDER",
            "owner": "ddm_cl1_capacity MAIN unsandboxed Metal executor",
            "consumer_store": "/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/",
            "blockers_observed_here": [
                "torch MPS is built but unavailable in the sandbox process",
                "the live claim registry has an active ddm_sd2 local_metal owner",
                "Terminal and Ghostty are blocked from Computer Use automation",
            ],
        },
        "pointer_moved": False,
        "exact_eval_fired": False,
    }
    hp.replace_json(args.output / "FINAL_RECEIPT.json", result)
    hp.replace_json(REPO_RECEIPT, result)
    state = json.loads(args.resume_from.read_text(encoding="utf-8"))
    state.update(
        {
            "complete": True,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "final_receipt": hp.file_record(args.output / "FINAL_RECEIPT.json"),
            "selected_candidate": selected_cell.name,
        }
    )
    hp.replace_json(args.resume_from, state)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--required-free-bytes", type=int, default=12 << 30)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "selected": result["full_n600"]["candidate"],
                "joint_delta_bytes": result["full_n600"]["delta_model_plus_raw_range_bytes_vs_base"],
                "archive_delta_bytes": result["full_n600"]["archive"]["delta_bytes_vs_exact_base"],
                "parity_exact": result["full_n600"]["frame_parity"]["byte_identical"],
                "exact_eval_fired": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
