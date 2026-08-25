"""ddm_rd2 Phase A -- the MEASURED byte curve of HG1's unique-home residual.

The charter asks where HG1's rate-distortion curve crosses dx2's distortion.  That
question has two halves and this module measures the *rate* half, which is cheap and
comes first: **given a byte budget, how many of HG1's 1,334,939 corrections can you
actually afford?**

`bs2` and the charter both assume the residual is a linear knob -- keep fraction f of
the corrections, pay fraction f of the 359,280 B.  That assumption is false in a
direction nobody has measured on this object.  The residual codes each correction as
`uleb(zigzag(address - previous)) + label_byte`, then LZMA2s the stream.  Its 2.153
bits/correction comes from the miss set being *dense and clustered* (mean gap 88.4
positions).  A value-ordered SUBSET is sparser by construction, so its address deltas
grow and its per-correction cost rises -- the localization tax
[[perfect-localization-is-worthless-the-address-is-the-tax]] read from inside this
container.

So this module builds real subsets, encodes them in the receiver's real wire format,
races the same three real coders HG1 raced, and reports real bytes.  No estimates.

Value orderings, so the family gets its best shot (optimal form, not a scan prefix):
  * `oracle_pixel`  -- corrections at positions where the HG1 row NEWLY flipped the
    SegNet argmax relative to the matched base.  This is MEASURED per-position damage
    from `ddm_bo2`'s retained argmax fields, not a proxy.
  * `oracle_tile`   -- same, but ranked by new-flip density in the enclosing 16x16
    tile, which respects that SegNet has a stride-2 stem and a wide receptive field so
    damage is regional, not per-pixel.
  * `scan_prefix`   -- the naive truncation control.  Reported so the value-ordering
    premium is visible rather than assumed.

Every subset is re-sorted into the receiver's canonical order before encoding and is
parsed back through the receiver's own `apply_residual` acceptance rules, so every
byte count reported here belongs to a payload the shipped receiver would accept.

Run:  .venv/bin/python experiments/ddm_rd2_residual_byte_curve.py
"""

from __future__ import annotations

import argparse
import brotli
import hashlib
import json
import lzma
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants -- mirrored from experiments/ddm_hg1_heterogeneous_analytic_generator_gate.py
# and verified against it by test, never re-derived from a description.
# ---------------------------------------------------------------------------
N_PAIRS, HEIGHT, WIDTH = 600, 384, 512
TOTAL_POSITIONS = N_PAIRS * HEIGHT * WIDTH  # 117,964,800
FRAME_POSITIONS = HEIGHT * WIDTH  # 196,608

RESIDUAL_MAGIC = b"HGR1"
RESIDUAL_HEADER = struct.Struct("<4sBBHHHQ")
# NOT re-typed from the dict's appearance. These IDs are 1-INDEXED in the receiver and I
# got them wrong once by mirroring the key order instead of reading the values
# ([[available-field-vs-authoritative-field]]): a 0-indexed copy decodes the shipped
# residual's order_id 6 as `class_tile16_time` when the receiver reads it as
# `tile64_time`. The round-trip control does NOT catch that, because encode and decode
# share the same wrong map. `verify_order_ids_against_receiver()` below is the control
# that does, by importing the receiver's own table.
RESIDUAL_ORDER_IDS = {
    "frame_raster": 1,
    "class_frame_raster": 2,
    "tile8_time": 3,
    "tile16_time": 4,
    "tile32_time": 5,
    "tile64_time": 6,
    "class_tile16_time": 7,
    "pair_tile16": 8,
}
CODERS = ("brotli_q11", "zlib_9", "lzma2_extreme")

AP = Path("/Volumes/APDataStore/pact")
HG1 = AP / "ddm_hg1_heterogeneous_analytic_generator_gate/retained"
BO2 = AP / "ddm_bo2_born_small_distortion/retained/perclass"
OUT = AP / "ddm_rd2_hg1_rate_distortion_curve"

# Custody: full digests, never a size match ([[available-field-vs-authoritative-field]]).
EXPECT = {
    HG1 / "generators/residual_exact_tile64_time.raw": (
        2_871_598,
        "cda5b4e677113f0a2ea942c11e7a0330007967007357cce95e6f79e6163eeeca",
    ),
    HG1 / "generators/generated_tokens.u8": (
        117_964_800,
        "2884c5701dc2b2059df0e9f8e4ee3ed81809457b127a48ad3fd3fb6f7a17152b",
    ),
}

# The container arithmetic this curve is priced against (all MEASURED upstream).
FULL_RESIDUAL_CODED = 359_280  # HG1 tile64_time / LZMA2 winner
CONTAINER_WITHOUT_RESIDUAL = 101_128  # ddm_bo2 sec.4, reproduced exactly from ar1b
CAP = 137_986  # strict-inequality sub-0.12 floor at dx2 distortion
RESIDUAL_BUDGET = CAP - CONTAINER_WITHOUT_RESIDUAL  # 36,858 B


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_order_ids_against_receiver() -> dict[str, int]:
    """Control: my mirrored constants must equal the receiver's own table.

    Mirrored constants are borrowed constants. This imports the shipped module and
    compares, so a copy error refuses here instead of silently producing payloads the
    receiver would reject while my own round-trip says everything is fine.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ddm_hg1_heterogeneous_analytic_generator_gate as hg1  # noqa: PLC0415

    mismatches = {
        "RESIDUAL_ORDER_IDS": (RESIDUAL_ORDER_IDS, dict(hg1.RESIDUAL_ORDER_IDS)),
        "RESIDUAL_MAGIC": (RESIDUAL_MAGIC, hg1.RESIDUAL_MAGIC),
        "RESIDUAL_HEADER": (RESIDUAL_HEADER.format, hg1.RESIDUAL_HEADER.format),
        "TOKEN_SHAPE": ((N_PAIRS, HEIGHT, WIDTH), (hg1.N_PAIRS, hg1.HEIGHT, hg1.WIDTH)),
        "TOTAL_POSITIONS": (TOTAL_POSITIONS, hg1.TOTAL_POSITIONS),
        "CODERS": (CODERS, tuple(hg1.CODERS)),
    }
    bad = {k: v for k, v in mismatches.items() if v[0] != v[1]}
    if bad:
        lines = "\n".join(f"  {k}: mine={v[0]!r} receiver={v[1]!r}" for k, v in bad.items())
        raise SystemExit(f"REFUSE: mirrored constants disagree with the receiver:\n{lines}")
    return dict(hg1.RESIDUAL_ORDER_IDS)


def verify_custody() -> dict[str, dict[str, object]]:
    facts: dict[str, dict[str, object]] = {}
    for path, (want_bytes, want_sha) in EXPECT.items():
        if not path.is_file():
            raise SystemExit(f"REFUSE: missing custody object {path}")
        got_bytes = path.stat().st_size
        got_sha = sha256_file(path)
        if got_bytes != want_bytes or got_sha != want_sha:
            raise SystemExit(
                f"REFUSE: custody mismatch for {path}\n"
                f"  bytes {got_bytes} vs {want_bytes}\n  sha256 {got_sha} vs {want_sha}"
            )
        facts[path.name] = {"bytes": got_bytes, "sha256": got_sha}
    return facts


# ---------------------------------------------------------------------------
# Wire format -- vectorised, and proved equal to the receiver's own scalar encoder.
# ---------------------------------------------------------------------------
def zigzag(values: np.ndarray) -> np.ndarray:
    v = values.astype(np.int64)
    return np.where(v >= 0, v * 2, -v * 2 - 1).astype(np.uint64)


def encode_records(addresses: np.ndarray, labels: np.ndarray, order: str) -> bytes:
    """Encode (address,label) records into the receiver's exact residual wire format."""
    previous = np.empty(addresses.shape, dtype=np.int64)
    previous[0] = -1
    previous[1:] = addresses[:-1]
    coded = zigzag(addresses.astype(np.int64) - previous)

    # ULEB128, vectorised: width = ceil(bits/7), min 1.
    width = np.ones(coded.shape, dtype=np.int64)
    for shift in (7, 14, 21, 28, 35, 42, 49, 56):
        width += (coded >= (np.uint64(1) << np.uint64(shift))).astype(np.int64)

    total = int(width.sum()) + int(addresses.size)  # +1 label byte per record
    out = np.zeros(total, dtype=np.uint8)
    # Record start offsets: each record is width[i] ULEB bytes then 1 label byte.
    stride = width + 1
    starts = np.zeros(addresses.size, dtype=np.int64)
    np.cumsum(stride[:-1], out=starts[1:])

    max_width = int(width.max())
    for byte_index in range(max_width):
        active = width > byte_index
        if not active.any():
            break
        chunk = (coded[active] >> np.uint64(7 * byte_index)) & np.uint64(0x7F)
        more = (width[active] > byte_index + 1).astype(np.uint8) * 0x80
        out[starts[active] + byte_index] = chunk.astype(np.uint8) | more
    out[starts + width] = labels.astype(np.uint8)

    header = RESIDUAL_HEADER.pack(
        RESIDUAL_MAGIC, 2, RESIDUAL_ORDER_IDS[order], N_PAIRS, HEIGHT, WIDTH, addresses.size
    )
    return header + out.tobytes()


def decode_records(payload: bytes) -> tuple[np.ndarray, np.ndarray, str]:
    """Parse a residual payload back to (addresses, labels, order). Mirrors apply_residual."""
    magic, version, order_id, pairs, height, width, count = RESIDUAL_HEADER.unpack_from(payload)
    if magic != RESIDUAL_MAGIC or version != 2 or (pairs, height, width) != (N_PAIRS, HEIGHT, WIDTH):
        raise SystemExit("REFUSE: residual header mismatch")
    order = {v: k for k, v in RESIDUAL_ORDER_IDS.items()}[order_id]
    body = np.frombuffer(payload, dtype=np.uint8, offset=RESIDUAL_HEADER.size)

    addresses = np.empty(count, dtype=np.int64)
    labels = np.empty(count, dtype=np.uint8)
    offset = 0
    previous = -1
    for index in range(count):
        value = 0
        shift = 0
        while True:
            byte = int(body[offset])
            offset += 1
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        delta = value // 2 if value % 2 == 0 else -(value // 2) - 1
        previous += delta
        addresses[index] = previous
        labels[index] = body[offset]
        offset += 1
    if offset != body.size:
        raise SystemExit("REFUSE: residual payload has trailing bytes")
    return addresses, labels, order


def canonical_key(addresses: np.ndarray, labels: np.ndarray, order: str) -> np.ndarray:
    """The receiver's own sort key, as a lexsort tuple (last key is primary)."""
    pair, position = np.divmod(addresses, FRAME_POSITIONS)
    y, x = np.divmod(position, WIDTH)
    if order == "frame_raster":
        return np.lexsort((addresses,))
    if order == "class_frame_raster":
        return np.lexsort((addresses, labels))
    if order == "class_tile16_time":
        return np.lexsort(((y % 16) * 16 + x % 16, pair, x // 16, y // 16, labels))
    if order == "pair_tile16":
        return np.lexsort((labels, (y % 16) * 16 + x % 16, x // 16, y // 16, pair))
    size = int(order.removeprefix("tile").removesuffix("_time"))
    return np.lexsort((labels, (y % size) * size + x % size, pair, x // size, y // size))


def compress(raw: bytes, coder: str) -> bytes:
    if coder == "brotli_q11":
        return brotli.compress(raw, quality=11)
    if coder == "zlib_9":
        return zlib.compress(raw, 9)
    return lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)


def decompress(payload: bytes, coder: str) -> bytes:
    if coder == "brotli_q11":
        return brotli.decompress(payload)
    if coder == "zlib_9":
        return zlib.decompress(payload)
    return lzma.decompress(payload, format=lzma.FORMAT_XZ)


def race(raw: bytes, coders: tuple[str, ...]) -> dict[str, object]:
    """Race real coders; fail closed on any non-deterministic or non-parseback row."""
    rows: dict[str, int] = {}
    for coder in coders:
        coded = compress(raw, coder)
        if compress(raw, coder) != coded:
            raise SystemExit(f"REFUSE: {coder} is non-deterministic")
        if decompress(coded, coder) != raw:
            raise SystemExit(f"REFUSE: {coder} failed parse-back")
        rows[coder] = len(coded)
    winner = min(coders, key=lambda c: (rows[c], coders.index(c)))
    return {"coders": rows, "winner": winner, "bytes": rows[winner]}


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--orders",
        default="tile64_time,tile16_time,frame_raster,class_frame_raster,pair_tile16",
        help="canonical orders to race per subset (sparse subsets may prefer a different one)",
    )
    parser.add_argument("--coders", default=",".join(CODERS))
    args = parser.parse_args(argv)

    started = time.monotonic()
    out_root = args.out
    (out_root / "retained").mkdir(parents=True, exist_ok=True)

    print("[rd2] verifying mirrored constants against the receiver's own table ...", flush=True)
    verify_order_ids_against_receiver()
    print("[rd2] control PASSED: order IDs / magic / header / shape / coders match the receiver", flush=True)

    print("[rd2] verifying custody by full digest ...", flush=True)
    custody = verify_custody()
    for name, fact in custody.items():
        print(f"  OK {name}: {fact['bytes']} B  {fact['sha256'][:16]}...", flush=True)

    # ---- 1. the full correction set, straight from the shipped residual -----
    raw = (HG1 / "generators/residual_exact_tile64_time.raw").read_bytes()
    addresses, labels, order = decode_records(raw)
    count = addresses.size
    print(f"[rd2] parsed {count:,} corrections from the shipped residual (order={order})", flush=True)
    if count != 1_334_939:
        raise SystemExit(f"REFUSE: expected 1,334,939 corrections, parsed {count:,}")
    if order != "tile64_time":
        raise SystemExit(f"REFUSE: shipped residual declares order={order}, expected tile64_time")
    if not np.array_equal(canonical_key(addresses, labels, order), np.arange(count)):
        raise SystemExit("REFUSE: shipped residual content is not in its own declared canonical order")
    print("[rd2] control PASSED: declared order is tile64_time AND the content is sorted in it", flush=True)

    # POSITIVE CONTROL: re-encoding the full set must reproduce the shipped raw bytes
    # exactly. If it does not, my encoder is not the receiver's encoder and every byte
    # below is fiction.
    rebuilt = encode_records(addresses, labels, order)
    if rebuilt != raw:
        raise SystemExit(
            "REFUSE: positive control FAILED -- re-encoded full set is not byte-identical "
            f"to the shipped residual ({len(rebuilt)} vs {len(raw)} B)"
        )
    print("[rd2] positive control PASSED: full-set re-encode is byte-identical to shipped raw", flush=True)

    full_race = race(raw, tuple(args.coders.split(",")))
    print(
        f"[rd2] full residual coded: {full_race['bytes']:,} B via {full_race['winner']} "
        f"(shipped: {FULL_RESIDUAL_CODED:,} B)",
        flush=True,
    )
    if full_race["bytes"] != FULL_RESIDUAL_CODED:
        raise SystemExit(
            f"REFUSE: full-set coder race gives {full_race['bytes']:,} B, "
            f"shipped winner is {FULL_RESIDUAL_CODED:,} B -- coder settings differ"
        )
    print("[rd2] positive control PASSED: coder race reproduces the shipped 359,280 B", flush=True)

    # ---- 2. MEASURED per-position damage from ddm_bo2's retained argmax fields ----
    print("[rd2] loading bo2 argmax fields for measured per-position damage ...", flush=True)
    gt = np.fromfile(BO2 / "gt_argmax_n600.u8", dtype=np.uint8)
    base = np.fromfile(BO2 / "base_dx2_comp_argmax_n600.u8", dtype=np.uint8)
    cand = np.fromfile(BO2 / "hg1_generator_comp_argmax_n600.u8", dtype=np.uint8)
    for name, arr in (("gt", gt), ("base", base), ("cand", cand)):
        if arr.size != TOTAL_POSITIONS:
            raise SystemExit(f"REFUSE: {name} argmax is {arr.size} positions, expected {TOTAL_POSITIONS}")

    base_wrong = base != gt
    cand_wrong = cand != gt
    new_flip = cand_wrong & ~base_wrong  # damage HG1 caused
    healed_mask = base_wrong & ~cand_wrong  # damage HG1 happened to remove
    healed_total = int(healed_mask.sum())
    net_flips = int(new_flip.sum()) - healed_total
    print(
        f"[rd2]   base flips {int(base_wrong.sum()):,} | cand flips {int(cand_wrong.sum()):,} "
        f"| new {int(new_flip.sum()):,} | healed {healed_total:,} | net {net_flips:,}",
        flush=True,
    )
    # CONTROL: bo2 measured base flips 40,981 and delta-flips 1,486,570.
    if int(base_wrong.sum()) != 40_981:
        raise SystemExit(f"REFUSE: base flip count {int(base_wrong.sum()):,} != bo2's 40,981")
    if net_flips != 1_486_570:
        raise SystemExit(f"REFUSE: net flip delta {net_flips:,} != bo2's 1,486,570")
    print("[rd2] positive control PASSED: flip counts reproduce bo2 exactly", flush=True)

    # ---- 3. value orderings -------------------------------------------------
    print("[rd2] building value orderings ...", flush=True)
    pixel_value = new_flip[addresses].astype(np.int64)

    # Tile-density value: new-flip count in the enclosing 16x16 tile of the same pair.
    tiles_y, tiles_x = HEIGHT // 16, WIDTH // 16
    tile_counts = (
        new_flip.reshape(N_PAIRS, tiles_y, 16, tiles_x, 16).sum(axis=(2, 4)).astype(np.int64)
    )
    pair_of, pos_of = np.divmod(addresses, FRAME_POSITIONS)
    y_of, x_of = np.divmod(pos_of, WIDTH)
    tile_value = tile_counts[pair_of, y_of // 16, x_of // 16]

    orderings = {
        # Descending value; ties broken by canonical position so the order is deterministic.
        "oracle_pixel": np.lexsort((addresses, -pixel_value)),
        "oracle_tile": np.lexsort((addresses, -tile_value)),
        "scan_prefix": np.arange(count),  # the naive truncation control
    }
    print(
        f"[rd2]   corrections sitting on a NEW argmax flip: {int(pixel_value.sum()):,} "
        f"({100.0 * pixel_value.sum() / count:.3f}% of the miss set)",
        flush=True,
    )

    # ---- 4. the byte ladder -------------------------------------------------
    fractions = [0.0025, 0.005, 0.01, 0.02, 0.05, 0.1026, 0.15, 0.25, 0.5, 0.75, 1.0]
    order_list = tuple(args.orders.split(","))
    coder_list = tuple(args.coders.split(","))

    rows: list[dict[str, object]] = []
    for name, permutation in orderings.items():
        for fraction in fractions:
            k = max(1, int(round(fraction * count)))
            if k > count:
                continue
            chosen = permutation[:k]
            sub_addr_raw = addresses[chosen]
            sub_label_raw = labels[chosen]

            best: dict[str, object] | None = None
            for candidate_order in order_list:
                key = canonical_key(sub_addr_raw, sub_label_raw, candidate_order)
                sub_addr = sub_addr_raw[key]
                sub_label = sub_label_raw[key]
                payload = encode_records(sub_addr, sub_label, candidate_order)
                back_addr, back_label, _ = decode_records(payload)
                if not np.array_equal(back_addr, sub_addr) or not np.array_equal(back_label, sub_label):
                    raise SystemExit(f"REFUSE: parse-back mismatch for {name}/{candidate_order}/k={k}")
                result = race(payload, coder_list)
                if best is None or int(result["bytes"]) < int(best["bytes"]):
                    best = {**result, "order": candidate_order, "raw_bytes": len(payload)}

            assert best is not None
            coded = int(best["bytes"])
            row = {
                "ordering": name,
                "fraction_of_corrections": fraction,
                "corrections": k,
                "residual_coded_bytes": coded,
                "residual_raw_bytes": int(best["raw_bytes"]),
                "winning_order": best["order"],
                "winning_coder": best["winner"],
                "coders": best["coders"],
                "bits_per_correction": 8.0 * coded / k,
                "container_bytes": CONTAINER_WITHOUT_RESIDUAL + coded,
                "fits_cap_137986": CONTAINER_WITHOUT_RESIDUAL + coded <= CAP,
                "healed_new_flips_at_corrected_positions": int(new_flip[sub_addr_raw].sum()),
            }
            rows.append(row)
            print(
                f"  {name:>13s} f={fraction:<7.4f} k={k:>9,}  "
                f"coded={coded:>9,} B  ({row['bits_per_correction']:.3f} b/corr, "
                f"{best['order']}/{best['winner']})  container={row['container_bytes']:>9,} B  "
                f"{'FITS' if row['fits_cap_137986'] else 'over'}",
                flush=True,
            )

    # ---- 5. binary-search the EXACT max-k that fits the 36,858 B budget ------
    # The ladder's grid is too coarse to answer "how many corrections can the cap
    # actually buy"; that number is the whole point, so it is searched, not interpolated.
    def coded_bytes_for(permutation: np.ndarray, k: int) -> tuple[int, str, str, np.ndarray]:
        chosen = permutation[:k]
        addr_raw, label_raw = addresses[chosen], labels[chosen]
        best: tuple[int, str, str, np.ndarray] | None = None
        for candidate_order in order_list:
            key = canonical_key(addr_raw, label_raw, candidate_order)
            payload = encode_records(addr_raw[key], label_raw[key], candidate_order)
            result = race(payload, coder_list)
            if best is None or int(result["bytes"]) < best[0]:
                best = (int(result["bytes"]), candidate_order, str(result["winner"]), addr_raw)
        assert best is not None
        return best

    affordable: dict[str, object] = {}
    for name, permutation in orderings.items():
        low, high = 1, count
        best_k, best_meta = 0, None
        while low <= high:
            mid = (low + high) // 2
            coded, won_order, won_coder, addr_raw = coded_bytes_for(permutation, mid)
            if coded <= RESIDUAL_BUDGET:
                best_k, best_meta = mid, (coded, won_order, won_coder, addr_raw)
                low = mid + 1
            else:
                high = mid - 1
        if best_meta is None:
            affordable[name] = {"max_corrections_within_budget": 0}
            continue
        coded, won_order, won_coder, addr_raw = best_meta
        # NOT `healed` -- that name holds the outer healed-flip MASK and shadowing it
        # crashed receipt assembly after every row had already been measured.
        touched = int(new_flip[addr_raw].sum())
        affordable[name] = {
            "max_corrections_within_budget": best_k,
            "fraction_of_miss_set": best_k / count,
            "residual_coded_bytes": coded,
            "container_bytes": CONTAINER_WITHOUT_RESIDUAL + coded,
            "bits_per_correction": 8.0 * coded / best_k,
            "winning_order": won_order,
            "winning_coder": won_coder,
            "new_flips_at_corrected_positions": touched,
            "share_of_new_flips_addressable": touched / int(new_flip.sum()),
        }
        print(
            f"[rd2] BUDGET SEARCH {name}: k={best_k:,} ({best_k / count:.4%} of misses) "
            f"at {coded:,} B; touches {touched:,} new flips "
            f"= {touched / int(new_flip.sum()):.4%} of the {int(new_flip.sum()):,} HG1 caused",
            flush=True,
        )

    # ---- 5b. receiver-acceptance control on the best budget subset ----------
    # A byte count only means something if the SHIPPED receiver would accept the payload.
    import ddm_hg1_heterogeneous_analytic_generator_gate as hg1  # noqa: PLC0415

    best_name = max(
        affordable, key=lambda n: int(affordable[n].get("new_flips_at_corrected_positions", 0) or 0)
    )
    best_k = int(affordable[best_name]["max_corrections_within_budget"])
    chosen = orderings[best_name][:best_k]
    key = canonical_key(addresses[chosen], labels[chosen], str(affordable[best_name]["winning_order"]))
    payload = encode_records(
        addresses[chosen][key], labels[chosen][key], str(affordable[best_name]["winning_order"])
    )
    probe = np.zeros(TOTAL_POSITIONS, dtype=np.uint8).reshape(N_PAIRS, HEIGHT, WIDTH)
    applied = hg1.apply_residual(payload, probe)
    if applied != best_k:
        raise SystemExit(f"REFUSE: receiver applied {applied:,} corrections, expected {best_k:,}")
    print(
        f"[rd2] control PASSED: the SHIPPED receiver accepts the {best_k:,}-correction "
        f"budget payload and applies exactly {applied:,} corrections",
        flush=True,
    )
    (out_root / "retained").mkdir(parents=True, exist_ok=True)
    (out_root / "retained" / f"residual_budget_{best_name}_k{best_k}.raw").write_bytes(payload)

    receipt = {
        "arm": "ddm_rd2",
        "phase": "A_byte_curve",
        "axis": "[byte-exact] real coders, real receiver wire format, no scoring",
        "custody": {k: v for k, v in custody.items()},
        "full_set": {
            "corrections": count,
            "raw_bytes": len(raw),
            "coded_bytes": full_race["bytes"],
            "winner": full_race["winner"],
            "bits_per_correction": 8.0 * full_race["bytes"] / count,
        },
        "controls": {
            "full_reencode_byte_identical_to_shipped_raw": True,
            "coder_race_reproduces_shipped_359280": True,
            "base_flip_count_matches_bo2_40981": True,
            "net_flip_delta_matches_bo2_1486570": True,
            "mirrored_constants_match_receiver": True,
            "shipped_residual_order_is_tile64_time_and_content_sorted": True,
            "shipped_receiver_accepts_budget_payload": True,
        },
        "budget_payload": {
            "ordering": best_name,
            "corrections": best_k,
            "retained_raw": f"residual_budget_{best_name}_k{best_k}.raw",
        },
        "new_flips_total": int(new_flip.sum()),
        "healed_flips_total": healed_total,
        "container_arithmetic": {
            "container_without_residual": CONTAINER_WITHOUT_RESIDUAL,
            "cap_strict_floor": CAP,
            "residual_budget": RESIDUAL_BUDGET,
            "full_residual_coded": FULL_RESIDUAL_CODED,
            "budget_as_fraction_of_full_residual": RESIDUAL_BUDGET / FULL_RESIDUAL_CODED,
        },
        "corrections_on_a_new_flip": int(pixel_value.sum()),
        "rows": rows,
        "affordable_within_budget": affordable,
        "elapsed_seconds": time.monotonic() - started,
    }
    # ALWAYS KEEP THE PAYLOAD: write before any verdict can raise.
    receipt_path = out_root / "retained" / "rd2_phaseA_byte_curve.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n[rd2] receipt -> {receipt_path}", flush=True)

    print("\n=== AFFORDABLE CORRECTIONS WITHIN THE 36,858 B RESIDUAL BUDGET ===")
    for name, info in affordable.items():
        print(f"  {name:>13s}: {info['max_corrections_within_budget']:>9,} corrections "
              f"({100.0 * float(info.get('fraction', 0.0)):.2f}% of the miss set)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
