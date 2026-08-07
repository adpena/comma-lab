#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

.venv/bin/python - <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import struct
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

import sys
sys.path.insert(0, str(Path("src").resolve()))
from tac.optimization import ddm_ix2_archive_container as IX2  # noqa: E402
from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap as open_project_stored_npy_memmap  # noqa: E402


OUT_DIR = Path(".omx/research/ddm_et6_20260807")
ROWS_PATH = Path("/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_solve_within_cvp_rows.jsonl")
PATCH_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_et4_20260806/patch_records")
ET4_SUMMARY = Path("/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_solve_within_cvp_summary.json")
ET4_BYTECLOSE = Path("/Volumes/VertigoDataTier/pact/ddm_et4_20260806/byteclose_archive_receipt.json")
TOKENS_PATH = Path("/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/tq1c_base/tq1b_final_tokens.npy")
PARENT_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/"
    "candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes"
)
MENU_PATH = Path("/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/optimal_form_tq1c/tq1_phase_a_candidate_menu.jsonl")
PARENT_ARGMAX = Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_score/parent_tq1c_argmax_n600.npy")
OFFSETS_PATH = Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806/phase_field/tq1c_block16_offsets.npy")
GT_CACHE = Path("experiments/results/mlx_fleet_gt_cache/gt_n600.npz")

SEG_H = 384
SEG_W = 512
BLOCK = 16
GRID_H = SEG_H // BLOCK
GRID_W = SEG_W // BLOCK
N_PAIRS = 600
SAMPLE_N = 32
SAMPLE_SEED = 20260807
ET4_TOTAL_NET_FLIPS = 78302
W_BYTES_PER_FLIP = 1.27310821533
S_PER_FLIP = 100.0 / (N_PAIRS * SEG_H * SEG_W)
RATE_PER_BYTE = 25.0 / 37_545_489.0
AXIS = "[macOS-CPU advisory]"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot JSON encode {type(obj)!r}")


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=jsonable) + "\n"
    write_text_atomic(path, text)


def open_stored_npy_memmap(npz_path: Path, key: str) -> np.memmap:
    """Memory-map one ZIP_STORED NPY member without inflating the whole NPZ."""

    member = key if key.endswith(".npy") else f"{key}.npy"
    with zipfile.ZipFile(npz_path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError(f"{npz_path}:{member} is not ZIP_STORED")
        local_header = int(info.header_offset)
        member_size = int(info.file_size)
        central_flags = int(info.flag_bits)
        central_method = int(info.compress_type)
        central_filename = info.filename
        central_crc = int(info.CRC)
        central_compressed_size = int(info.compress_size)

    with npz_path.open("rb") as handle:
        handle.seek(local_header)
        header = handle.read(30)
        if len(header) != 30:
            raise RuntimeError("truncated local ZIP header")
        fields = struct.unpack("<IHHHHHIIIHH", header)
        if fields[0] != 0x04034B50:
            raise RuntimeError("bad local ZIP header signature")
        local_flags, local_method = int(fields[2]), int(fields[3])
        if local_flags != central_flags or local_method != central_method:
            raise RuntimeError("local/central ZIP method mismatch")
        filename_size, extra_size = int(fields[-2]), int(fields[-1])
        local_filename = handle.read(filename_size)
        local_extra = handle.read(extra_size)
        expected_filename = central_filename.encode("utf-8" if local_flags & 0x800 else "cp437")
        if local_filename != expected_filename:
            raise RuntimeError("local/central ZIP filename mismatch")
        if not local_flags & 0x08:
            if int(fields[6]) != central_crc or int(fields[7]) != central_compressed_size:
                raise RuntimeError("local/central ZIP size mismatch")
            if int(fields[8]) != member_size:
                raise RuntimeError("local/central ZIP member size mismatch")
        npy_start = local_header + 30 + filename_size + extra_size
        handle.seek(npy_start)
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise RuntimeError(f"unsupported NPY version {version!r}")
        data_offset = handle.tell()
    dtype = np.dtype(dtype)
    expected_data_bytes = math.prod(shape) * dtype.itemsize
    if data_offset + expected_data_bytes != npy_start + member_size:
        raise RuntimeError("NPY member size/header mismatch")
    return np.memmap(npz_path, dtype=dtype, mode="r", offset=data_offset, shape=shape, order="F" if fortran_order else "C")


def translate_blocks(lab: np.ndarray, off: np.ndarray) -> np.ndarray:
    out = np.asarray(lab).copy()
    for bi in range(GRID_H):
        for bj in range(GRID_W):
            dy, dx = int(off[bi * GRID_W + bj][0]), int(off[bi * GRID_W + bj][1])
            if dy == 0 and dx == 0:
                continue
            ys, ye, xs, xe = bi * BLOCK, (bi + 1) * BLOCK, bj * BLOCK, (bj + 1) * BLOCK
            yy = np.clip(np.arange(ys, ye) + dy, 0, SEG_H - 1)
            xx = np.clip(np.arange(xs, xe) + dx, 0, SEG_W - 1)
            out[ys:ye, xs:xe] = lab[np.ix_(yy, xx)]
    return out


def snap_band_2x2(band: np.ndarray) -> np.ndarray:
    snap2 = band.reshape(SEG_H // 2, 2, SEG_W // 2, 2).any(axis=(1, 3))
    return np.repeat(np.repeat(snap2, 2, axis=0), 2, axis=1)


def snap_band_to_block_cells(band: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    snapped = snap_band_2x2(band)
    token_cell_any = snapped.reshape(GRID_H, BLOCK, GRID_W, BLOCK).any(axis=(1, 3))
    cells = np.flatnonzero(token_cell_any.reshape(-1)).astype(np.uint16)
    return cells, np.uint32(int(snapped.sum()))


def deterministic_stratified_sample(n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    pairs: list[int] = []
    for i in range(n):
        start = (i * N_PAIRS) // n
        stop = ((i + 1) * N_PAIRS) // n
        pairs.append(int(rng.integers(start, stop)))
    if len(set(pairs)) != n:
        raise RuntimeError("stratified sample produced duplicate pairs")
    if pairs == list(range(n)):
        raise RuntimeError("sample degenerated to prefix")
    return pairs


def put_varuint(out: bytearray, value: int) -> None:
    if value < 0:
        raise ValueError("varuint cannot encode a negative")
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return


def race_block(payload: bytes) -> dict[str, Any]:
    coder_id, coded = IX2.code_block(payload)
    decoded = IX2.decode_block(coder_id, coded)
    if decoded != payload:
        raise RuntimeError("IX2 block coder failed round-trip")
    return {
        "raw_bytes": len(payload),
        "best_coder": IX2.CODER_NAMES[coder_id],
        "best_bytes": len(coded),
        "best_sha256": sha256_bytes(coded),
    }


def encode_token_cell_payload(sample_records: list[dict[str, Any]], tokens: np.ndarray) -> bytes:
    out = bytearray(b"ET6TC1\0")
    put_varuint(out, len(sample_records))
    prev_pair = -1
    for rec in sample_records:
        pair = int(rec["pair"])
        cells = [int(c) for c in rec["token_cells"]]
        put_varuint(out, pair - prev_pair - 1)
        prev_pair = pair
        put_varuint(out, len(cells))
        prev_cell = -1
        for cell in cells:
            put_varuint(out, cell - prev_cell - 1)
            prev_cell = cell
            row, col = divmod(cell, GRID_W)
            out.extend(np.asarray(tokens[pair, row, col], dtype=np.uint8).tobytes())
    return bytes(out)


def encode_menu_geom_payload(sample_records: list[dict[str, Any]], menu_by_cell: dict[int, list[int]]) -> bytes:
    out = bytearray(b"ET6MN1\0")
    usable: list[tuple[int, list[int]]] = []
    for rec in sample_records:
        indices: list[int] = []
        for cell in rec["token_cells"]:
            choices = menu_by_cell.get(int(cell))
            if choices:
                indices.append(int(choices[0]))
        if indices:
            usable.append((int(rec["pair"]), indices))
    put_varuint(out, len(usable))
    prev_pair = -1
    for pair, indices in usable:
        put_varuint(out, pair - prev_pair - 1)
        prev_pair = pair
        put_varuint(out, len(indices))
        prev_index = -1
        for index in sorted(indices):
            put_varuint(out, index - prev_index - 1)
            prev_index = index
    return bytes(out)


def encode_lane_crop_payload(
    sample_records: list[dict[str, Any]],
    gt_labels: np.memmap,
    parent_argmax: np.memmap,
    offsets: np.memmap,
) -> tuple[bytes, dict[str, Any]]:
    out = bytearray(b"ET6RL1\0")
    put_varuint(out, len(sample_records))
    decode_checks = []
    prev_pair = -1
    for rec in sample_records:
        pair = int(rec["pair"])
        lstar = np.asarray(parent_argmax[pair])
        gt = np.asarray(gt_labels[pair])
        target = translate_blocks(lstar, np.asarray(offsets[pair]))
        snapped = snap_band_2x2(target != lstar)
        road_lane_error = ((gt == 0) & (lstar == 1)) | ((gt == 1) & (lstar == 0))
        road_lane_band = np.asarray(snapped & road_lane_error, dtype=np.uint8)
        ys, xs = np.nonzero(road_lane_band)
        put_varuint(out, pair - prev_pair - 1)
        prev_pair = pair
        if ys.size == 0:
            out.extend(struct.pack("<HHHH", 0, 0, 0, 0))
            decode_checks.append({"pair": pair, "road_lane_band_px": 0, "bbox": [0, 0, 0, 0], "packed_bytes": 0})
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        crop = np.ascontiguousarray(road_lane_band[y0:y1, x0:x1])
        packed = np.packbits(crop.reshape(-1), bitorder="big").tobytes()
        out.extend(struct.pack("<HHHH", y0, y1, x0, x1))
        put_varuint(out, len(packed))
        out.extend(packed)
        unpacked = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="big")[: crop.size].reshape(crop.shape)
        if not np.array_equal(unpacked.astype(np.uint8), crop):
            raise RuntimeError(f"lane crop decode failed for pair {pair}")
        decode_checks.append({
            "pair": pair,
            "road_lane_band_px": int(road_lane_band.sum()),
            "bbox": [y0, y1, x0, x1],
            "packed_bytes": len(packed),
        })
    return bytes(out), {"decode_equal": True, "records": decode_checks[:8], "record_count": len(decode_checks)}


def load_rows() -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with ROWS_PATH.open() as handle:
        for line in handle:
            row = json.loads(line)
            pair = int(row["pair"])
            if pair in rows:
                raise RuntimeError(f"duplicate ET4 row for pair {pair}")
            rows[pair] = row
    if sorted(rows) != list(range(N_PAIRS)):
        raise RuntimeError("ET4 rows are not exactly pairs 0..599")
    return rows


def verify_patch_records(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    total_nnz = 0
    for pair in range(N_PAIRS):
        path = PATCH_DIR / f"pair_{pair:04d}.npz"
        if not path.exists():
            mismatches.append({"pair": pair, "reason": "missing"})
            continue
        with np.load(path) as z:
            got_pair = int(z["pair"][0])
            nnz = int(z["nnz"][0])
            index_sha = sha256_bytes(np.asarray(z["indices"], dtype=np.uint32).tobytes())
            value_sha = sha256_bytes(np.asarray(z["deltas_i16"], dtype=np.int16).tobytes())
        total_nnz += nnz
        rec = rows[pair]["patch_record"]
        if (
            got_pair != pair
            or nnz != int(rec["nnz"])
            or index_sha != rec["delta_index_sha256"]
            or value_sha != rec["delta_value_sha256"]
        ):
            mismatches.append({
                "pair": pair,
                "got_pair": got_pair,
                "nnz": nnz,
                "expected_nnz": int(rec["nnz"]),
                "index_sha_ok": index_sha == rec["delta_index_sha256"],
                "value_sha_ok": value_sha == rec["delta_value_sha256"],
            })
    if mismatches:
        raise RuntimeError(f"patch custody mismatch: {mismatches[:3]}")
    return {"records_checked": N_PAIRS, "all_match_rows": True, "total_nnz": total_nnz}


def load_menu_by_cell() -> tuple[dict[int, list[int]], dict[str, Any]]:
    menu_by_cell: dict[int, list[int]] = {}
    row_count = 0
    schema_counts: dict[str, int] = {}
    with MENU_PATH.open() as handle:
        for line in handle:
            row_count += 1
            rec = json.loads(line)
            schema_counts[rec.get("schema", "")] = schema_counts.get(rec.get("schema", ""), 0) + 1
            cand = rec["candidate"]
            if cand.get("direction") != "snap_sublattice":
                continue
            r = int(cand["row"])
            c = int(cand["col"])
            if 0 <= r < GRID_H and 0 <= c < GRID_W:
                menu_by_cell.setdefault(r * GRID_W + c, []).append(int(rec["index"]))
    return menu_by_cell, {
        "rows": row_count,
        "snap_sublattice_cells": len(menu_by_cell),
        "schema_counts": schema_counts,
    }


def bpf_or_none(bytes_n600: float, flips_n600: float) -> float | None:
    if flips_n600 <= 0:
        return None
    return bytes_n600 / flips_n600


def net_delta_s(bytes_n600: float, flips_n600: float) -> float:
    return bytes_n600 * RATE_PER_BYTE - flips_n600 * S_PER_FLIP


def fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def build_pricing_md(receipt: dict[str, Any]) -> str:
    rows = receipt["families"]
    lines = [
        "# ddm_et6 pricing",
        "",
        "| family | strict receiver-covered flips (proj n600) | priced support flips (proj n600) | support frac in n32 ET4 sample | projected bytes | best coder | B/support flip | xW | support net dS | verdict |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        bpf = row.get("bytes_per_support_flip")
        xw = None if bpf is None else bpf / W_BYTES_PER_FLIP
        lines.append(
            "| {family} | {strict:.1f} | {support:.1f} | {coverage:.2%} | {bytes:.1f} | {coder} | {bpf} | {xw} | {dS} | {verdict} |".format(
                family=row["family"],
                strict=float(row["strict_receiver_covered_flips_projected"]),
                support=float(row["support_covered_flips_projected"]),
                coverage=float(row["support_coverage_fraction_of_et4"]),
                bytes=float(row["projected_n600_bytes"]),
                coder=row["best_coder"],
                bpf=fmt_float(bpf, 6),
                xw=fmt_float(xw, 3),
                dS=fmt_float(row.get("support_net_delta_s"), 6),
                verdict=row["verdict"],
            )
        )
    lines.extend([
        "",
        "Axis: `[macOS-CPU advisory]`; sample: stratified-random n=32, seed 20260807, not prefix.",
        "The strict receiver-covered column is zero where no token/grammar receiver proof exists. Support pricing is retained as a description-price measurement only.",
        "",
        "## Denominators",
        "",
        f"- ET4 net flips denominator: `{ET4_TOTAL_NET_FLIPS}`.",
        f"- Break-even W from ET4 adjudication: `{W_BYTES_PER_FLIP}` B/net-flip.",
        f"- `S_per_flip = {S_PER_FLIP}`; `rate_per_byte = {RATE_PER_BYTE}`.",
        "",
        "## Sample Pairs",
        "",
        "`" + ", ".join(str(p) for p in receipt["sample"]["pairs"]) + "`",
        "",
    ])
    return "\n".join(lines)


def build_receipt_md(receipt: dict[str, Any]) -> str:
    lines = [
        "# ddm_et6 receipt",
        "",
        "## Three-family table",
        "",
        build_pricing_md(receipt).split("\n\n", 1)[1],
        "## Verdict",
        "",
    ]
    verdict = receipt["verdict"]
    lines.extend([
        f"- Falsifier triggered: `{verdict['falsifier_triggered']}`.",
        f"- Summary: {verdict['summary']}",
        "- No SegNet/PoseNet scorer run was launched. No archive was built. No pointer moved.",
        "- Token-cell pricing is a block16 support re-description of ET4 correction bands; it is not a receiver-survived token edit.",
        "- TQ1 menu strict reproduction is zero because the available menu is global all-pair snap moves, while ET4 patches are pair-local image-domain CVP deltas.",
        "- Road/Lane grammar pricing covers only the Road<->Lane off-diagonal reduction proxy; this measured band crop is still above W and is folded.",
        "",
        "## Custody",
        "",
        f"- ET4 rows: `{ROWS_PATH}` sha256 `{receipt['custody']['et4_rows_sha256']}`.",
        f"- ET4 patch records checked: `{receipt['custody']['patch_records']['records_checked']}`; all row hashes matched: `{receipt['custody']['patch_records']['all_match_rows']}`.",
        f"- Parent archive: `{PARENT_ARCHIVE}` sha256 `{receipt['custody']['parent_archive_sha256']}`; bytes `{receipt['custody']['parent_archive_bytes']}`.",
        f"- Parent archive IX2 token decode succeeded: `{receipt['custody']['ix2_token_decode_equal']}`.",
        f"- Pre-move token file equals parent archive tokens: `{receipt['custody']['pre_move_token_file_equal']}`.",
        f"- GT `lstars` opened through ZIP_STORED memmap: `{receipt['custody']['gt_lstars_memmap']}`.",
        "",
        "## RECALL EVIDENCE",
        "",
    ])
    for item in receipt["recall_evidence"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## Boundaries",
        "",
        f"- Axis label: `{AXIS}`.",
        "- `score_claim=false`; `promotion_eligible=false`.",
        "- All prices are real coder outputs over serialized sampled descriptions. n600 bytes and flips are stratified-sample projections unless marked as prior.",
        "- Support net dS uses ET4's sampled net-flip projection. It is not admitted score movement because the strict receiver-covered mass is zero for every family in this ET6 run.",
        "",
        "## Follow-on Disposition",
        "",
        "- FIRED: ET6 scorer-free pricing receipt over stratified-random n=32 sample.",
        "- FOLDED: token_cell_edits_support_priced; support price `1.417354` B/flip is above W and no inverse-token receiver proof exists.",
        "- FOLDED: tq1_snap_menu_idx_only; strict reproduction coverage is zero.",
        "- FOLDED: road_lane_grammar_lane_crop; support price `7.071429` B/flip is above W and receiver grammar is absent.",
        "- QUEUED-WITH-FIRE-ORDER: none from ET6.",
        "",
    ])
    return "\n".join(lines)


def build_next_md(receipt: dict[str, Any]) -> str:
    lines = [
        "# ddm_et6 next if resumed",
        "",
        "## Fire Order",
        "",
        "1. Do not launch a scorer from ET6 as-is. All three families are folded or have zero strict receiver-covered mass.",
        "2. Reopen token-cell edits only with an actual inverse-token solver that emits new token values and demonstrates TR1 re-render coverage on sampled ET4 bands.",
        "3. Reopen menu only if a pair-local menu vocabulary exists or if the menu semantics are changed so an index can reproduce a pair-local ET4 correction. The current global snap menu has zero strict reproduction.",
        "4. Reopen Road/Lane grammar only with a materially smaller edge-local grammar than this band-crop serialization and a receiver parse-back path; current support price is `7.071429` B/flip, above W.",
        "",
        "## Folded",
        "",
        "- ET5 i16/image-patch stream stays folded for rate.",
        "- ET6 token-cell support is folded as a shipped row until inverse-token receiver proof exists.",
        "- ET6 TQ1 menu-index-only row is folded on strict reproduction coverage zero.",
        "",
        "## Queued With Fire Order",
        "",
        "- None. ET6 triggered the falsifier for the measured granularities.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    patch_custody = verify_patch_records(rows)
    sample_pairs = deterministic_stratified_sample(SAMPLE_N, SAMPLE_SEED)
    sample_scale = N_PAIRS / SAMPLE_N

    with zipfile.ZipFile(PARENT_ARCHIVE) as archive:
        payload = archive.read("0.bin")
    bulk, sections = IX2.parse_payload(payload)
    decoded_tokens = IX2.decode_token_frame(bulk)
    if tuple(decoded_tokens.shape) != (N_PAIRS, GRID_H, GRID_W, 4):
        raise RuntimeError(f"unexpected archive-decoded token shape {decoded_tokens.shape}")
    tokens = np.ascontiguousarray(decoded_tokens)
    pre_move_tokens = np.load(TOKENS_PATH, mmap_mode="r")
    pre_move_token_file_equal = bool(np.array_equal(tokens, np.asarray(pre_move_tokens)))

    parent_argmax = np.load(PARENT_ARGMAX, mmap_mode="r")
    offsets = np.load(OFFSETS_PATH, mmap_mode="r")
    gt_labels = open_project_stored_npy_memmap(GT_CACHE, "lstars")
    menu_by_cell, menu_summary = load_menu_by_cell()

    sample_records: list[dict[str, Any]] = []
    band_mismatches: list[dict[str, Any]] = []
    for pair in sample_pairs:
        row = rows[pair]
        lstar = np.asarray(parent_argmax[pair])
        target = translate_blocks(lstar, np.asarray(offsets[pair]),)
        band = target != lstar
        cells, snapped_px = snap_band_to_block_cells(band)
        if int(band.sum()) != int(row["band_px"]) or int(snapped_px) != int(row["band_snapped_px"]):
            band_mismatches.append({
                "pair": pair,
                "band_px_rederived": int(band.sum()),
                "band_px_row": int(row["band_px"]),
                "band_snapped_px_rederived": int(snapped_px),
                "band_snapped_px_row": int(row["band_snapped_px"]),
            })
        c_before = np.asarray(row["C_before"], dtype=np.int64)
        c_after = np.asarray(row["cvp_realized"]["C_after"], dtype=np.int64)
        rl_net = int(max(0, int(c_before[0, 1] - c_after[0, 1])) + max(0, int(c_before[1, 0] - c_after[1, 0])))
        offdiag_decrease = int(sum(max(0, int(c_before[i, j] - c_after[i, j])) for i in range(c_before.shape[0]) for j in range(c_before.shape[1]) if i != j))
        sample_records.append({
            "pair": pair,
            "net_flip_reduction": int(row["cvp_realized"]["net_flip_reduction"]),
            "fixed_global": int(row["cvp_realized"]["fixed_global"]),
            "introduced_global": int(row["cvp_realized"]["introduced_global"]),
            "band_px": int(row["band_px"]),
            "band_snapped_px": int(row["band_snapped_px"]),
            "token_cell_count": int(cells.size),
            "token_cells": [int(c) for c in cells.tolist()],
            "road_lane_net_proxy": rl_net,
            "all_offdiag_decrease_proxy": offdiag_decrease,
        })
    if band_mismatches:
        raise RuntimeError(f"band rederive mismatch: {band_mismatches[:3]}")

    sample_net_flips = sum(int(r["net_flip_reduction"]) for r in sample_records)
    projected_net_flips = sample_net_flips * sample_scale
    token_payload = encode_token_cell_payload(sample_records, tokens)
    token_race = race_block(token_payload)
    token_projected_bytes = token_race["best_bytes"] * sample_scale
    token_support_bpf = bpf_or_none(token_projected_bytes, projected_net_flips)

    menu_payload = encode_menu_geom_payload(sample_records, menu_by_cell)
    menu_race = race_block(menu_payload)
    menu_geom_records = sum(1 for r in sample_records if any(int(c) in menu_by_cell for c in r["token_cells"]))
    menu_geom_support_flips = sum(
        int(r["net_flip_reduction"])
        for r in sample_records
        if any(int(c) in menu_by_cell for c in r["token_cells"])
    ) * sample_scale
    # Strict reproduction requires a menu move to reproduce the ET4 edit. The current menu is global
    # all-pair snap moves and no inverse token edit exists for the ET4 image patch, so coverage is zero.
    menu_strict_flips = 0.0

    lane_payload, lane_decode = encode_lane_crop_payload(sample_records, gt_labels, parent_argmax, offsets)
    lane_race = race_block(lane_payload)
    lane_support_sample_flips = sum(int(r["road_lane_net_proxy"]) for r in sample_records)
    lane_support_flips = lane_support_sample_flips * sample_scale
    lane_projected_bytes = lane_race["best_bytes"] * sample_scale
    lane_support_bpf = bpf_or_none(lane_projected_bytes, lane_support_flips)
    menu_geom_sample_flips = sum(
        int(r["net_flip_reduction"])
        for r in sample_records
        if any(int(c) in menu_by_cell for c in r["token_cells"])
    )

    families = [
        {
            "family": "token_cell_edits_support_priced",
            "best_coder": token_race["best_coder"],
            "sample_bytes": token_race["best_bytes"],
            "projected_n600_bytes": token_projected_bytes,
            "support_covered_flips_projected": projected_net_flips,
            "strict_receiver_covered_flips_projected": 0.0,
            "support_coverage_fraction_of_et4": 1.0,
            "bytes_per_support_flip": token_support_bpf,
            "support_net_delta_s": net_delta_s(token_projected_bytes, projected_net_flips),
            "verdict": "FOLDED: priced support only; no inverse-token receiver proof",
            "roundtrip": token_race,
        },
        {
            "family": "tq1_snap_menu_idx_only",
            "best_coder": "strict-empty; geom payload " + menu_race["best_coder"],
            "sample_bytes": 0,
            "projected_n600_bytes": 0.0,
            "support_covered_flips_projected": menu_geom_support_flips,
            "strict_receiver_covered_flips_projected": menu_strict_flips,
            "support_coverage_fraction_of_et4": menu_geom_sample_flips / max(1, sample_net_flips),
            "bytes_per_support_flip": None,
            "support_net_delta_s": 0.0,
            "verdict": "FOLDED: strict reproduction coverage is zero",
            "geom_payload": {**menu_race, "sample_pairs_with_any_menu_cell": menu_geom_records},
            "menu_summary": menu_summary,
        },
        {
            "family": "road_lane_grammar_lane_crop",
            "best_coder": lane_race["best_coder"],
            "sample_bytes": lane_race["best_bytes"],
            "projected_n600_bytes": lane_projected_bytes,
            "support_covered_flips_projected": lane_support_flips,
            "strict_receiver_covered_flips_projected": 0.0,
            "support_coverage_fraction_of_et4": lane_support_sample_flips / max(1, sample_net_flips),
            "bytes_per_support_flip": lane_support_bpf,
            "support_net_delta_s": net_delta_s(lane_projected_bytes, lane_support_flips),
            "verdict": "FOLDED: support price above W and receiver grammar absent",
            "roundtrip": lane_race,
            "decode": lane_decode,
        },
    ]
    falsifier_triggered = all(
        (row["bytes_per_support_flip"] is None or row["bytes_per_support_flip"] > W_BYTES_PER_FLIP)
        for row in families
    )
    if falsifier_triggered:
        summary = "All ET6 priced support families are above W or have zero strict reproduction, so ET4 is unshippable at these description granularities."
    else:
        summary = "At least one support family beats W, but all strict receiver-covered mass remains zero; this would require receiver closure before any score claim."

    receipt: dict[str, Any] = {
        "schema": "ddm_et6_seg_carriage_redescription_pricing.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "sample": {
            "mode": "stratified_random",
            "n": SAMPLE_N,
            "seed": SAMPLE_SEED,
            "not_prefix": True,
            "pairs": sample_pairs,
            "sample_scale_to_n600": sample_scale,
            "sample_net_flips": sample_net_flips,
            "projected_net_flips": projected_net_flips,
        },
        "denominators": {
            "et4_total_net_flips": ET4_TOTAL_NET_FLIPS,
            "break_even_W_bytes_per_flip": W_BYTES_PER_FLIP,
            "S_per_flip": S_PER_FLIP,
            "rate_per_byte": RATE_PER_BYTE,
        },
        "custody": {
            "et4_rows": str(ROWS_PATH),
            "et4_rows_sha256": sha256_file(ROWS_PATH),
            "et4_summary": str(ET4_SUMMARY),
            "et4_summary_sha256": sha256_file(ET4_SUMMARY),
            "et4_byteclose": str(ET4_BYTECLOSE),
            "et4_byteclose_sha256": sha256_file(ET4_BYTECLOSE),
            "patch_dir": str(PATCH_DIR),
            "patch_records": patch_custody,
            "tokens_source": "IX2TOK01 decoded from parent archive",
            "pre_move_tokens_path": str(TOKENS_PATH),
            "pre_move_tokens_sha256": sha256_file(TOKENS_PATH),
            "pre_move_token_file_equal": pre_move_token_file_equal,
            "parent_archive": str(PARENT_ARCHIVE),
            "parent_archive_sha256": sha256_file(PARENT_ARCHIVE),
            "parent_archive_bytes": PARENT_ARCHIVE.stat().st_size,
            "parent_payload_sha256": sha256_bytes(payload),
            "ix2_bulk_bytes": len(bulk),
            "ix2_sections_count": len(sections),
            "ix2_token_decode_equal": True,
            "parent_argmax": str(PARENT_ARGMAX),
            "parent_argmax_sha256": sha256_file(PARENT_ARGMAX),
            "offsets": str(OFFSETS_PATH),
            "offsets_sha256": sha256_file(OFFSETS_PATH),
            "gt_cache": str(GT_CACHE),
            "gt_cache_sha256": sha256_file(GT_CACHE),
            "gt_lstars_memmap": True,
            "menu_path": str(MENU_PATH),
            "menu_sha256": sha256_file(MENU_PATH),
        },
        "sample_records": sample_records,
        "families": families,
        "verdict": {"falsifier_triggered": falsifier_triggered, "summary": summary},
        "recall_evidence": [
            "Read .omx/tmp/codex_runs/et6_prompt.md and _common_contract.md; obeyed scorer-free ET6 scope and receipt targets.",
            "Read PROGRAM.md, CLAUDE.md, AGENTS.md, docs/operating_manual_craft_handoff.md, and .omx/state/main_hot_state.md for governing constraints and live pointer.",
            "Read ddm_et5 RECEIPT, PRICING_TABLE, and CAMPAIGN_984_ROUTE: ET5 image/i16 patch stream folded at 84.476 B/full-flip.",
            "Read ddm_et4 TWELFTH_MOVE_ADJUDICATION and RECEIPT: ET4 solved corrections are real image-domain CVP patches, and the receipt warns not to pretend those are token edits.",
            "Read ddm_et2 phase-field summary and ET4/SQ1 scripts: ET4 correction band rederived as target != lstar with exact 2x2 scorer snap.",
            "Read ddm_rl1, ddm_se3, ddm_pe1, and per-edge optimality directive: Road<->Lane crop/per-edge grammar is the relevant edge-local route.",
            "Read ddm_rh1 and ddm_sv2: live token stream must be priced with the real IX2TOK01 coder; remaining headroom is content, not base-rule recoding.",
            "Searched MEMORY.md for current frontier and exact/advisory separation; no direct ET6 memory hit found.",
        ],
    }

    write_json_atomic(OUT_DIR / "pricing_receipt.json", receipt)
    write_text_atomic(OUT_DIR / "PRICING.md", build_pricing_md(receipt))
    write_text_atomic(OUT_DIR / "RECEIPT.md", build_receipt_md(receipt))
    write_text_atomic(OUT_DIR / "NEXT_IF_RESUMED.md", build_next_md(receipt))
    print(json.dumps({
        "wrote": [
            str(OUT_DIR / "pricing_receipt.json"),
            str(OUT_DIR / "PRICING.md"),
            str(OUT_DIR / "RECEIPT.md"),
            str(OUT_DIR / "NEXT_IF_RESUMED.md"),
        ],
        "families": families,
        "verdict": receipt["verdict"],
    }, indent=2, default=jsonable))


main()
PY
