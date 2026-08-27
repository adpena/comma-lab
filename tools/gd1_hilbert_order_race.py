#!/usr/bin/env python
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""ddm_gd1 QA82 scan-order race: Hilbert vs raster serialization of the token lattice.

Operator escalation (2026-07-31, "Maybe hilbert curve"): the SMEVR token coder scans the
24x32 cell lattice in RASTER order (experiments/ddm_r7_token_coder.py::_encode_smevr,
`row, column = divmod(cell, width)`) — a textbook-default scan order per the
generic-default law (memory: generic_basis_metric_never_optimal_cosine_fourier_euclid).
This race re-serializes the SAME sealed payload through the SAME landed r7 coder in
alternative cell orders and measures real lossless bytes.

Scorer-free by construction: pure re-order + re-code of existing archive payloads.
No SegNet/PoseNet, no Metal, no GPU — safe while the QA24 burn holds the scorer slot.

Arms (same-coder same-payload discipline; r7 harness REUSED, never rebuilt):
  a0_smevr_2d      control: canonical 2D raster SMEVR (must reproduce archive bytes)
  a1_smevr_raster1 (P,1,N,C) raster linearization — isolates the 2D-context loss
  a2_smevr_hilbert (P,1,N,C) gilbert2d(32x24) linearization — the operator's arm
  a3_smevr_serpent (P,1,N,C) boustrophedon linearization — cheap adjacency control
  b*_brotli11 / c*_lzma1: generic coders on packed nibbles, 2D vs Hilbert order
Keep-mask race (24x32 bool, 384 kept): raster/hilbert/serpentine bit order through an
order-1 adaptive binary arithmetic coder + zlib-9 + brotli-11.

Every SMEVR/brotli/lzma token arm is decode-verified lossless and inverse-permutation
checked against the original codes. All rows [macOS-CPU advisory, rate-only];
score_claim=false; pointer 0.1910828242 [contest-CPU] UNMOVED.

Receipt: .omx/research/ddm_gd1_hilbert_order_race_receipt_20260731.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np

from tac.race_receipt import (
    RivalryRow,
    rivalry_rows_from_arms,
    unadopted_better_challengers,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from ddm_r7_token_coder import (  # noqa: E402  (r7 landed custody, reused)
    _ArithmeticDecoder,
    _ArithmeticEncoder,
    _decode_token_codes,
    encode_token_codes,
)

ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/"
    "v4d_composed_refine_celldrop50_archive.zip"
)
KEEP_MASK = Path("/Volumes/VertigoDataTier/pact/ddm_sg1_20260731/qa24_grid_keep_mask_50.npy")
RECEIPT = REPO / ".omx/research/ddm_gd1_hilbert_order_race_receipt_20260731.json"

GRID_H, GRID_W = 24, 32  # cell lattice of the [600,24,32,4] token field


def _sgn(value: int) -> int:
    return (value > 0) - (value < 0)


def _gilbert_generate(x, y, ax, ay, bx, by):
    """Generalized Hilbert curve for arbitrary rectangles (Cerveny gilbert2d)."""
    w = abs(ax + ay)
    h = abs(bx + by)
    dax, day = _sgn(ax), _sgn(ay)
    dbx, dby = _sgn(bx), _sgn(by)
    if h == 1:
        for _ in range(w):
            yield (x, y)
            x, y = x + dax, y + day
        return
    if w == 1:
        for _ in range(h):
            yield (x, y)
            x, y = x + dbx, y + dby
        return
    ax2, ay2 = ax // 2, ay // 2
    bx2, by2 = bx // 2, by // 2
    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)
    if 2 * w > 3 * h:
        if (w2 % 2) and (w > 2):
            ax2, ay2 = ax2 + dax, ay2 + day
        yield from _gilbert_generate(x, y, ax2, ay2, bx, by)
        yield from _gilbert_generate(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)
    else:
        if (h2 % 2) and (h > 2):
            bx2, by2 = bx2 + dbx, by2 + dby
        yield from _gilbert_generate(x, y, bx2, by2, ax2, ay2)
        yield from _gilbert_generate(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
        yield from _gilbert_generate(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            -bx2,
            -by2,
            -(ax - ax2),
            -(ay - ay2),
        )


def hilbert_cell_order(height: int, width: int) -> np.ndarray:
    """Cell visit order (flat raster indices) along a gilbert2d curve; unit-adjacent."""
    if width >= height:
        pts = list(_gilbert_generate(0, 0, width, 0, 0, height))
    else:
        pts = list(_gilbert_generate(0, 0, 0, height, width, 0))
    order = np.asarray([y * width + x for (x, y) in pts], dtype=np.int64)
    if sorted(order.tolist()) != list(range(height * width)):
        raise AssertionError("gilbert order is not a permutation")
    steps = np.abs(np.diff(np.asarray([p[0] for p in pts]))) + np.abs(
        np.diff(np.asarray([p[1] for p in pts]))
    )
    if not bool(np.all(steps == 1)):
        raise AssertionError("gilbert order has non-unit steps")
    return order


def serpentine_cell_order(height: int, width: int) -> np.ndarray:
    rows = []
    for r in range(height):
        cols = range(width) if r % 2 == 0 else range(width - 1, -1, -1)
        rows.extend(r * width + c for c in cols)
    return np.asarray(rows, dtype=np.int64)


def linearize(codes: np.ndarray, order: np.ndarray) -> np.ndarray:
    pairs, height, width, channels = codes.shape
    flat = codes.reshape(pairs, height * width, channels)
    return np.ascontiguousarray(flat[:, order, :]).reshape(pairs, 1, height * width, channels)


def race_token_arm(name, codes_arr, codec, results, original, order=None):
    t0 = time.time()
    frame = encode_token_codes(codes_arr, levels=16, codec=codec)
    decoded = _decode_token_codes(frame, canonical=False)
    if not np.array_equal(decoded, codes_arr):
        raise AssertionError(f"{name}: decode differs — arm is NOT lossless")
    if order is not None:
        pairs, height, width, channels = original.shape
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        restored = decoded.reshape(pairs, height * width, channels)[:, inverse, :]
        if not np.array_equal(restored.reshape(original.shape), original):
            raise AssertionError(f"{name}: inverse permutation does not restore payload")
    results[name] = {"bytes": len(frame), "codec": codec, "seconds": round(time.time() - t0, 2)}
    print(f"  {name:<22} {len(frame):>9,} B  ({results[name]['seconds']}s)", flush=True)


def adaptive_binary_bytes(bits: np.ndarray) -> int:
    """Order-2 adaptive binary arithmetic coder (context = previous two bits)."""
    enc = _ArithmeticEncoder()
    rows: dict[int, list[int]] = {}
    prev1 = prev2 = 0
    for bit in bits.tolist():
        ctx = prev1 * 2 + prev2
        counts = rows.setdefault(ctx, [1, 1])
        enc.encode(bit, counts)
        counts[bit] += 1
        prev2, prev1 = prev1, bit
    payload = enc.finish()
    dec = _ArithmeticDecoder(payload)
    rows2: dict[int, list[int]] = {}
    prev1 = prev2 = 0
    for bit in bits.tolist():
        ctx = prev1 * 2 + prev2
        counts = rows2.setdefault(ctx, [1, 1])
        got = dec.decode(counts)
        if got != bit:
            raise AssertionError("mask AC round-trip differs")
        counts[got] += 1
        prev2, prev1 = prev1, bit
    return len(payload)


def race_mask(mask: np.ndarray) -> dict:
    try:
        import brotli
    except ImportError:  # pragma: no cover
        brotli = None
    flat = mask.reshape(-1).astype(np.uint8)
    orders = {
        "raster": np.arange(flat.size, dtype=np.int64),
        "hilbert": hilbert_cell_order(GRID_H, GRID_W),
        "serpentine": serpentine_cell_order(GRID_H, GRID_W),
    }
    out = {"raw_bytes": int((flat.size + 7) // 8)}
    for oname, order in orders.items():
        bits = flat[order]
        packed = np.packbits(bits).tobytes()
        row = {
            "adaptive_binary_ac": adaptive_binary_bytes(bits),
            "zlib9": len(zlib.compress(packed, 9)),
        }
        if brotli is not None:
            row["brotli11"] = len(brotli.compress(packed, quality=11))
        out[oname] = row
    return out


# Named explicitly rather than matched by name prefix: a future "a4_" arm that changed the
# coder would silently join the SCAN-ORDER role and compare two factors at once.
_SCAN_ORDER_ARMS = ("a0_smevr_2d_control", "a1_smevr_raster_1d", "a2_smevr_hilbert_1d",
                    "a3_smevr_serpent_1d")
_LOST = ("refused: lost the measured race against the live incumbent on the same sealed payload "
         "through the same coder — adopting it would raise bytes")
_AXIS = "[macOS-CPU advisory, rate-only lossless byte measurement]"


def _rivalry_rows(results: dict[str, dict], mask_rows: dict[str, dict]) -> list[RivalryRow]:
    """Record WHO RACED WHOM, per gd5's §5 schema, split so no row varies two factors at once.

    The arms answer three different one-slot questions and must not be pooled: a1-a3 change the
    SCAN ORDER through the same SMEVR coder, b0/c0 change the CODER at the same 2D order, and the
    keep-mask arms change the bit order of a different payload entirely. A single flat "everything
    vs a0" block would look tidier and would compare a scan order against a coder swap.
    """
    scan = rivalry_rows_from_arms(
        role="token_scan_order",
        incumbent_arm="a0_smevr_2d_control",
        arms={k: v for k, v in results.items() if k in _SCAN_ORDER_ARMS},
        value_key="bytes", metric="lossless coded bytes of state/tokens.dr7t",
        axis=_AXIS, unit="B", not_adopted_reason=_LOST,
        notes="incumbent = raster order in experiments/ddm_r7_token_coder.py::_encode_smevr",
    )
    coder = rivalry_rows_from_arms(
        role="token_entropy_coder",
        incumbent_arm="a0_smevr_2d_control",
        arms={k: v for k, v in results.items()
              if k in ("a0_smevr_2d_control", "b0_brotli11_2d", "c0_lzma1_2d")},
        value_key="bytes", metric="lossless coded bytes of state/tokens.dr7t",
        axis=_AXIS, unit="B", not_adopted_reason=_LOST,
        notes="same 2D order, different coder — isolates the coder from the scan order",
    )
    mask_arms = {k: v for k, v in mask_rows.items() if isinstance(v, dict)}
    mask = rivalry_rows_from_arms(
        role="keep_mask_bit_order",
        incumbent_arm="raster",
        arms=mask_arms, value_key="adaptive_binary_ac",
        metric="adaptive binary AC bytes of the 24x32 keep mask",
        axis=_AXIS, unit="B", not_adopted_reason=_LOST,
        notes="incumbent = raster bit order of the QA24 keep mask",
    )
    return scan + coder + mask


def main() -> None:
    print("[gd1] loading sealed payload", flush=True)
    with zipfile.ZipFile(ARCHIVE) as zf:
        token_frame = zf.read("state/tokens.dr7t")
    codes = _decode_token_codes(token_frame, canonical=False)
    payload_sha = hashlib.sha256(token_frame).hexdigest()
    print(f"  tokens.dr7t {len(token_frame):,} B sha {payload_sha[:12]} shape {codes.shape}")

    hilbert = hilbert_cell_order(GRID_H, GRID_W)
    serpent = serpentine_cell_order(GRID_H, GRID_W)
    raster = np.arange(GRID_H * GRID_W, dtype=np.int64)

    results: dict[str, dict] = {}
    print("[gd1] SMEVR arms", flush=True)
    race_token_arm("a0_smevr_2d_control", codes, "smevr", results, codes)
    race_token_arm("a1_smevr_raster_1d", linearize(codes, raster), "smevr", results, codes, raster)
    race_token_arm("a2_smevr_hilbert_1d", linearize(codes, hilbert), "smevr", results, codes, hilbert)
    race_token_arm("a3_smevr_serpent_1d", linearize(codes, serpent), "smevr", results, codes, serpent)
    print("[gd1] generic-coder arms", flush=True)
    race_token_arm("b0_brotli11_2d", codes, "brotli11", results, codes)
    race_token_arm("b1_brotli11_hilbert", linearize(codes, hilbert), "brotli11", results, codes, hilbert)
    race_token_arm("c0_lzma1_2d", codes, "lzma1", results, codes)
    race_token_arm("c1_lzma1_hilbert", linearize(codes, hilbert), "lzma1", results, codes, hilbert)

    print("[gd1] keep-mask arms", flush=True)
    mask_rows = race_mask(np.load(KEEP_MASK))
    print(f"  {json.dumps(mask_rows)}")

    rivalry = _rivalry_rows(results, mask_rows)
    unadopted_winners = unadopted_better_challengers(rivalry)

    stored = len(token_frame)
    receipt = {
        "schema": "ddm_gd1_hilbert_order_race.v2",
        "axis": "[macOS-CPU advisory, rate-only lossless byte measurement]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "payload": {
            "archive": str(ARCHIVE),
            "member": "state/tokens.dr7t",
            "stored_bytes": stored,
            "sha256": payload_sha,
            "shape": list(codes.shape),
        },
        "coder_custody": "experiments/ddm_r7_token_coder.py (r7 landed harness, reused)",
        "token_arms": results,
        "keep_mask_arms": mask_rows,
        "rivalry": [r.to_dict() for r in rivalry],
        "rivalry_schema": "tac.race_receipt.RivalryRow (ddm_wt1, task #868)",
        "rivalry_excluded_arms": {
            "b1_brotli11_hilbert": "varies BOTH coder and scan order vs the live path — a "
                                   "two-factor arm cannot answer a one-slot rivalry question",
            "c1_lzma1_hilbert": "varies BOTH coder and scan order vs the live path",
        },
        "verification": "every token arm decode-verified lossless + inverse-permutation checked",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    print(f"[gd1] receipt -> {RECEIPT}")
    print(
        f"[gd1] rivalry: {len(rivalry)} row(s) recorded across "
        f"{len({r.role for r in rivalry})} role(s); "
        f"{len(unadopted_winners)} of {len(rivalry)} are unadopted challengers that BEAT the "
        "live incumbent"
    )
    for row in unadopted_winners:
        print(f"  ! {row.role}: {row.challenger} beats {row.incumbent} by {row.delta:+g} "
              f"{row.unit} — {row.not_adopted_reason}")
    control = results["a0_smevr_2d_control"]["bytes"]
    print(
        f"[gd1] verdict inputs: control {control:,} B (stored {stored:,}) | "
        f"hilbert {results['a2_smevr_hilbert_1d']['bytes']:,} B | "
        f"raster-1d {results['a1_smevr_raster_1d']['bytes']:,} B"
    )


if __name__ == "__main__":
    main()
