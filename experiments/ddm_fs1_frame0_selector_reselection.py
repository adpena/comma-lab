"""ddm_fs1 -- byte-close the frame-0 selector re-selection ddm_pr1 measured, on afr1.

WHY THIS ARM EXISTS
-------------------
``ddm_pr1`` Sec 12.1 swept all 8 frame-0 selector modes over all 600 pairs of the LIVE
afr1 body and found 39 pairs whose best mode beats their shipped mode by more than 1%
-- including pair 85, whose shipped selector op is ACTIVELY HARMFUL on the current
renders.  Priced through the receiver's own blob formula the whole adoption is
**+36 B for a net -1.032e-04 S** ``[macOS-CPU advisory projection]``.  pr1 could not
ship it: the shipped runtime is decode-only.  This module writes the encoder, splices
the archive the receiver's way, and proves the result byte-for-byte.

THE SPLICE IS SMALLER THAN ``ddm_up3``'s, AND THE REASON IS MEASURED
--------------------------------------------------------------------
``ddm_up3`` had to run the whole CAP1 stack backwards because it changed the carrier
CODES.  This arm changes only the selector, and the selector tail is INVARIANT under
every layer between the brotli stream and the receiver's ``decode_selector``:

* ``rr5_arith_basis.split_carrier_body`` / ``assemble_carrier_body``
  (rr5_arith_basis.py:374-441) carry ``body_tail = body[packed_portion:]`` through
  verbatim, and ``restore_carrier_body`` (:499) rebuilds it unchanged;
* ``dx2_cabac_coefficients.restore_carrier_body`` (:319-330) uses the same split and
  likewise never touches ``body_tail``;
* ``residual_archive._decode_rx1_models`` (:210-247) then derives
  ``_packed_portion`` from the body's own u24 bit counts and reads the selector as
  ``SPARSE_SELECTOR_PREFIX + carrier_body[cap1_bytes:]``, and
  ``102 + 80 == _cap1_body_bytes``'s fixed part, so the tail it lifts is exactly the
  stored ``body_tail``.

That invariance is ASSERTED at build time on the shipped body (``control_tail``), not
assumed: the splice refuses if the stored tail and the fully-restored tail differ.  So
the whole byte-close is "replace the tail, re-compress, re-header, re-zip", and the
container identity control below proves the encoder reproduces the shipped archive
bit-for-bit when the tail is unchanged.

THE CONTAINER IS HELD FIXED ON PURPOSE
--------------------------------------
The brotli parameters are DISCOVERED by identity search over the shipped carrier
stream (q=9, lgwin=16 reproduces it exactly) and then held.  A different quality or
window might make the archive smaller, but that byte would be a CONTAINER gain, not
the selector's, and mixing it into this delta would break the one-variable comparison
([[m103]]).  ``build --container-search`` measures the alternatives and reports them
as a separate, orthogonal number; the headline candidate always uses the shipped shape.

AUTHORITY
---------
Every number here is bytes, or ``[macOS-CPU advisory]`` frozen CPU-torch PoseNet on
DALI-lineage GT.  ``score_claim=false``, ``promotable=false``.  Only
``upstream/evaluate.py`` on contest hardware, on these exact archive bytes, is a score.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

from tac.semantic_pipeline.frame0_selector_codec import (
    FRAME_COUNT,
    STORED_PREFIX,
    encode_selector,
    selector_blob_length,
    stored_tail,
)

#: The afr1 frontier body, sha-gated so a delta can never be taken against a
#: superseded object ([[m103]]).
FRONTIER_ARCHIVE_SHA256 = (
    "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25"
)
FRONTIER_ARCHIVE_BYTES = 180_002

#: The afr1 contest-CUDA T4 n600 receipt legs (call fc-01M1C2ZZQEQWNE0FT06R3WZJCS,
#: ``ddm_afr1_tile48_groupbin8_cuda_n600_20260831``).  Recomputed from components at
#: every use; the report's 2-dp ``Final score`` line is never read (#877).
AFR1_D_SEG_T4 = 0.00020139
AFR1_D_POSE_T4 = 6.37e-06
AFR1_SCORE_T4 = 0.14797617125559104

#: ``upstream/evaluate.py:63,90`` -- rate = archive bytes / uncompressed source bytes.
UNCOMPRESSED_SOURCE_BYTES = 37_545_489

#: DISCOVERED by identity search over the shipped carrier stream and then ASSERTED by
#: ``control_identity`` on every build.  Not assumed.
BROTLI_QUALITY = 9
BROTLI_LGWIN = 16

#: Encoder-only container alternatives, measured for reporting only.  The shipped
#: shape is index 0 and always wins the headline.
CONTAINER_OPTIONS: tuple[tuple[int, int], ...] = (
    (BROTLI_QUALITY, BROTLI_LGWIN),
    (9, 24),
    (10, 16),
    (10, 24),
    (11, 16),
    (11, 22),
    (11, 24),
)

AXIS_ADVISORY = "[macOS-CPU advisory, frozen CPU-torch PoseNet]"


class Fs1Error(RuntimeError):
    """A ddm_fs1 precondition failed.  Fail closed, never approximate."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def pose_leg(d_pose: float) -> float:
    return math.sqrt(10.0 * d_pose)


def composed_score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    """``upstream/evaluate.py:90`` recomputed from components, never the 2-dp display."""
    return (
        100.0 * d_seg
        + pose_leg(d_pose)
        + 25.0 * archive_bytes / UNCOMPRESSED_SOURCE_BYTES
    )


# --------------------------------------------------------------------------
# The shipped container, read out of the receiver rather than guessed.
# --------------------------------------------------------------------------


def import_runtime(runtime_dir: Path):
    """Import ``residual_archive`` + the RR5/DX2 riders from ONE runtime tree.

    Runtime-bound rather than recalled: the splice must use the exact algorithms
    carried by the archive it rebuilds, not another generation's copy.
    """
    runtime_dir = Path(runtime_dir).resolve()
    sys.path.insert(0, str(runtime_dir))
    try:
        # Purge only the ``runtime`` PACKAGE, never every module whose name merely
        # begins with those seven letters: a bare prefix test would also evict an
        # unrelated ``runtime_foo`` a caller had imported.
        for stale in [
            name
            for name in sys.modules
            if name == "runtime" or name.startswith("runtime.")
        ]:
            del sys.modules[stale]
        from runtime import (
            dx2_cabac_coefficients,  # type: ignore[import-not-found]
            frame0_selector,  # type: ignore[import-not-found]
            residual_archive,  # type: ignore[import-not-found]
            rr5_arith_basis,  # type: ignore[import-not-found]
        )
    finally:
        sys.path.pop(0)
    return residual_archive, rr5_arith_basis, dx2_cabac_coefficients, frame0_selector


class ShippedBody:
    """Every field of the shipped archive the splice needs, all derived from lengths."""

    def __init__(self, archive_path: Path, runtime_dir: Path):
        import brotli

        self.archive_path = Path(archive_path).resolve()
        self.runtime_dir = Path(runtime_dir).resolve()
        self.archive_bytes_raw = self.archive_path.read_bytes()
        self.archive_size = len(self.archive_bytes_raw)
        self.archive_sha256 = sha256_bytes(self.archive_bytes_raw)
        ra, rr5, dx2, f0s = import_runtime(self.runtime_dir)
        self.ra, self.rr5, self.dx2, self.f0s = ra, rr5, dx2, f0s

        with zipfile.ZipFile(io.BytesIO(self.archive_bytes_raw)) as archive:
            if archive.namelist() != ["p"]:
                raise Fs1Error("archive must contain exactly member p")
            info = archive.getinfo("p")
            self.zip_info = {
                "date_time": list(info.date_time),
                "compress_type": int(info.compress_type),
                "create_system": int(info.create_system),
                "external_attr": int(info.external_attr),
            }
            self.outer = archive.read("p")

        header = ra.RX1_MODEL_HEADER.unpack_from(self.outer)
        if header[0] != ra.RX1_MAGIC:
            raise Fs1Error("fs1 only splices the RX1 container the afr1 body uses")
        self.rx1_header = header
        (
            _magic,
            _version,
            _codec,
            _table_mode,
            self.reserved,
            hpac_bytes,
            semantic_bytes,
            carrier_bytes,
        ) = header
        offset = ra.RX1_MODEL_HEADER.size
        self.hpac_stream = self.outer[offset : offset + hpac_bytes]
        offset += hpac_bytes
        self.semantic_stream = self.outer[offset : offset + semantic_bytes]
        offset += semantic_bytes
        self.carrier_stream = self.outer[offset : offset + carrier_bytes]
        offset += carrier_bytes
        self.section_tail = self.outer[offset:]

        if self.reserved & ra.CK2_RESERVED_CARRIER_PLANE2:
            raise Fs1Error(
                "this body sets the CK2 carrier plane-2 interleave; fs1's tail splice "
                "has only been proven on the afr1 body, which does not"
            )
        self.stored_carrier_body = brotli.decompress(self.carrier_stream)
        fields = rr5.split_carrier_body(self.stored_carrier_body)
        self.packed_portion = len(self.stored_carrier_body) - len(
            bytes(fields["body_tail"])
        )
        self.stored_tail_bytes = bytes(fields["body_tail"])

        restored = self.stored_carrier_body
        if self.reserved & ra.RR5_RESERVED_ARITH_BASIS:
            restored = rr5.restore_carrier_body(restored)
        if self.reserved & ra.DX2_RESERVED_CABAC_COEFFICIENTS:
            restored = dx2.restore_carrier_body(restored)
        packed = (
            102
            + 40
            + ((int.from_bytes(restored[0:3], "little") + 7) // 8)
            + ((int.from_bytes(restored[3:6], "little") + 7) // 8)
        )
        restored = ra._restore_packed_cap1_metadata(restored[:packed]) + restored[packed:]
        cap1_bytes = ra._cap1_body_bytes(restored)
        self.receiver_tail_bytes = restored[cap1_bytes:]
        if self.receiver_tail_bytes != self.stored_tail_bytes:
            raise Fs1Error(
                "the selector tail is NOT invariant through the RR5/DX2/packed-CAP1 "
                "layers on this body; the tail splice is unproven here and refuses"
            )
        self.selector_blob = STORED_PREFIX + self.stored_tail_bytes
        _modes, choices = f0s.decode_selector(self.selector_blob)
        self.selector_choices = np.asarray(choices, dtype=np.uint8)

    def facts(self) -> dict[str, Any]:
        return {
            "archive_path": str(self.archive_path),
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_size,
            "runtime_dir": str(self.runtime_dir),
            "rx1_reserved": f"{self.reserved:#06x}",
            "stored_carrier_body_bytes": len(self.stored_carrier_body),
            "carrier_stream_bytes": len(self.carrier_stream),
            "selector_blob_bytes": len(self.selector_blob),
            "selector_blob_sha256": sha256_bytes(self.selector_blob),
            "selector_active_pairs": np.flatnonzero(self.selector_choices).tolist(),
            "selector_active_modes": self.selector_choices[
                np.flatnonzero(self.selector_choices)
            ].tolist(),
            "selector_tail_invariant_through_riders": True,
        }


def write_archive(body: ShippedBody, tail: bytes, *, quality: int, lgwin: int) -> bytes:
    """Rebuild ``archive.zip`` with ``tail`` in the selector slot and nothing else changed."""
    import brotli

    new_body = body.stored_carrier_body[: body.packed_portion] + tail
    stream = brotli.compress(new_body, quality=quality, lgwin=lgwin)
    if brotli.decompress(stream) != new_body:
        raise Fs1Error(f"brotli round-trip failed at q={quality} lgwin={lgwin}")
    magic, version, codec, table_mode, reserved, hpac_bytes, semantic_bytes, _old = (
        body.rx1_header
    )
    outer = b"".join(
        (
            body.ra.RX1_MODEL_HEADER.pack(
                magic,
                version,
                codec,
                table_mode,
                reserved,
                hpac_bytes,
                semantic_bytes,
                len(stream),
            ),
            body.hpac_stream,
            body.semantic_stream,
            stream,
            body.section_tail,
        )
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED, allowZip64=False) as archive:
        entry = zipfile.ZipInfo("p", date_time=tuple(body.zip_info["date_time"]))
        entry.compress_type = body.zip_info["compress_type"]
        entry.create_system = body.zip_info["create_system"]
        entry.external_attr = body.zip_info["external_attr"]
        archive.writestr(entry, outer)
    return buffer.getvalue()


def control_identity(body: ShippedBody) -> dict[str, Any]:
    """Rebuild with the UNCHANGED tail; the bytes must equal the shipped archive.

    This is the no-op detector for the whole container path: brotli parameters, RX1
    header packing, ZIP framing.  If it fails, no delta this module reports means
    anything, so ``build`` refuses on it.
    """
    rebuilt = write_archive(
        body, body.stored_tail_bytes, quality=BROTLI_QUALITY, lgwin=BROTLI_LGWIN
    )
    identical = rebuilt == body.archive_bytes_raw
    return {
        "identical_to_shipped_archive": bool(identical),
        "rebuilt_bytes": len(rebuilt),
        "rebuilt_sha256": sha256_bytes(rebuilt),
        "shipped_bytes": body.archive_size,
        "shipped_sha256": body.archive_sha256,
        "brotli_quality": BROTLI_QUALITY,
        "brotli_lgwin": BROTLI_LGWIN,
    }


def parse_back_parts(archive_bytes: bytes, runtime_dir: Path) -> dict[str, Any]:
    """Parse candidate bytes through the SHIPPED receiver; return section digests.

    The parse is the receiver's own ``read_residual_archive``, so a candidate that
    this function accepts is one the contest runtime accepts, and every returned
    digest is the digest of a section the receiver actually consumes.
    """
    ra, _rr5, _dx2, f0s = import_runtime(runtime_dir)
    sys.path.insert(0, str(Path(runtime_dir).resolve()))
    try:
        from runtime.carrier_repack import (  # type: ignore[import-not-found]
            split_frame0_selector_carrier,
        )
    finally:
        sys.path.pop(0)
    with tempfile.TemporaryDirectory(prefix="fs1_parse_") as scratch:
        staged = Path(scratch) / "archive.zip"
        staged.write_bytes(archive_bytes)
        parts = ra.read_residual_archive(staged)
    carrier_blob, selector_blob = split_frame0_selector_carrier(parts.carrier_blob)
    if selector_blob is None:
        raise Fs1Error("the parsed archive carries no frame-0 selector")
    _modes, choices = f0s.decode_selector(selector_blob)
    return {
        "semantic_sha256": sha256_bytes(parts.semantic_blob),
        "semantic_bytes": len(parts.semantic_blob),
        "carrier_cap1_sha256": sha256_bytes(carrier_blob),
        "carrier_cap1_bytes": len(carrier_blob),
        "hpac_sha256": sha256_bytes(parts.hpac_blob),
        "hpac_bytes": len(parts.hpac_blob),
        "token_stream_sha256": sha256_bytes(parts.token_stream),
        "token_stream_bytes": len(parts.token_stream),
        "residual_payload_sha256": sha256_bytes(parts.residual_payload),
        "table_codes_sha256": sha256_array(parts.table.codes),
        "table_scale": float(parts.table.scale),
        "compensation_blob": None
        if parts.compensation_blob is None
        else sha256_bytes(parts.compensation_blob),
        "selector_blob_sha256": sha256_bytes(selector_blob),
        "selector_blob_bytes": len(selector_blob),
        "selector_choices": np.asarray(choices, dtype=np.uint8),
    }


# --------------------------------------------------------------------------
# mode=select -- read pr1's sweep, apply the margin gate, price the adoption.
# --------------------------------------------------------------------------


def _admissible_rows(
    sweep: dict[str, Any], *, margin: float, shipped_choices: np.ndarray
) -> list[dict[str, Any]]:
    """Rows that improve and clear the robustness gate, checked against THIS body."""
    rows = sweep["rows"]
    if len(rows) != FRAME_COUNT:
        raise Fs1Error(f"sweep covers {len(rows)} pairs, need all {FRAME_COUNT}")
    admissible = []
    for row in rows:
        pair = int(row["pair"])
        if int(row["shipped_mode"]) != int(shipped_choices[pair]):
            raise Fs1Error(
                f"sweep pair {pair} records shipped mode {row['shipped_mode']} but the "
                f"archive carries {int(shipped_choices[pair])}; the sweep was taken on "
                "a different body and refuses"
            )
        if row["gain"] <= 0.0:
            continue
        if margin > 1.0 and not row["ratio"] > margin:
            continue
        admissible.append(row)
    return admissible


def adoption_from_sweep(
    sweep: dict[str, Any],
    *,
    margin: float,
    shipped_choices: np.ndarray,
    strategy: str = "ratio_gate",
    base_d_pose: float | None = None,
) -> dict[str, Any]:
    """Turn pr1's per-pair sweep into one adopted selector, under one of two rules.

    ``margin`` is the ROBUSTNESS gate on ``d_pose(shipped) / d_pose(best)``.  It
    exists because the sweep is batch 1 while the score is batch 8: pr1 measured the
    batch-shape spread at well under 1%, so a >1% margin survives the cross-shape
    step and a 0.1% one may not.  ``margin = 1.0`` admits every positive gain.

    ``strategy`` then decides how much of the admissible set to buy:

    ``ratio_gate``
        adopt EVERY admissible row.  This is ``ddm_pr1`` Sec 12.1's pre-registered
        rule and reproduces its +36 B / -1.032e-04 exactly.

    ``byte_optimal``
        the blob length is a function of the ACTIVE COUNT alone and the per-pair
        gains are additive, so for each achievable ``k`` the best set is the top-``k``
        admissible rows by gain, and sweeping ``k`` gives the exact net-dS frontier
        rather than a heuristic.  Rows that keep ``k`` fixed (mode -> other mode) or
        LOWER it (mode -> identity, i.e. pair 85) are always taken: they carry gain at
        zero or negative byte cost.  Requires ``base_d_pose`` to price the pose leg.
    """
    if strategy not in ("ratio_gate", "byte_optimal"):
        raise Fs1Error(f"unknown adoption strategy {strategy!r}")
    admissible = _admissible_rows(
        sweep, margin=margin, shipped_choices=shipped_choices
    )
    shipped_active = int(np.count_nonzero(shipped_choices))
    frontier: list[dict[str, Any]] = []
    if strategy == "ratio_gate":
        adopted = list(admissible)
    else:
        if base_d_pose is None:
            raise Fs1Error("byte_optimal adoption needs base_d_pose to price the frontier")
        free = [r for r in admissible if r["shipped_mode"] != 0 and r["best_mode"] != 0]
        lowering = [r for r in admissible if r["shipped_mode"] != 0 and r["best_mode"] == 0]
        raising = sorted(
            (r for r in admissible if r["shipped_mode"] == 0),
            key=lambda r: -float(r["gain"]),
        )
        floor_gain = sum(float(r["gain"]) for r in free + lowering)
        floor_k = shipped_active - len(lowering)
        best_index, best_net, cumulative = 0, None, 0.0
        for index in range(len(raising) + 1):
            if index:
                cumulative += float(raising[index - 1]["gain"])
            active = floor_k + index
            gain = floor_gain + cumulative
            net = (
                pose_leg(base_d_pose - gain / FRAME_COUNT)
                - pose_leg(base_d_pose)
                + 25.0
                * (selector_blob_length(active) - selector_blob_length(shipped_active))
                / UNCOMPRESSED_SOURCE_BYTES
            )
            frontier.append(
                {
                    "raising_taken": index,
                    "active_pairs": active,
                    "blob_bytes": selector_blob_length(active),
                    "delta_bytes": selector_blob_length(active)
                    - selector_blob_length(shipped_active),
                    "total_d_pose_gain": gain,
                    "net_delta_S": net,
                }
            )
            if best_net is None or net < best_net:
                best_index, best_net = index, net
        adopted = free + lowering + raising[:best_index]

    choices = np.asarray(shipped_choices, dtype=np.uint8).copy()
    changed: list[dict[str, Any]] = []
    gain = 0.0
    for row in sorted(adopted, key=lambda r: int(r["pair"])):
        pair = int(row["pair"])
        choices[pair] = np.uint8(int(row["best_mode"]))
        gain += float(row["gain"])
        changed.append(
            {
                "pair": pair,
                "from_mode": int(row["shipped_mode"]),
                "to_mode": int(row["best_mode"]),
                "d_pose_from": float(row["d_pose_at_shipped_mode"]),
                "d_pose_to": float(row["d_pose_at_best_mode"]),
                "gain": float(row["gain"]),
                "ratio": float(row["ratio"]),
            }
        )
    active = int(np.count_nonzero(choices))
    return {
        "strategy": strategy,
        "margin_gate": margin,
        "admissible_rows": len(admissible),
        "byte_frontier": frontier,
        "min_adopted_ratio": min((c["ratio"] for c in changed), default=None),
        "changed_pairs": changed,
        "changed_count": len(changed),
        "active_pairs": active,
        "shipped_active_pairs": shipped_active,
        "newly_active": sum(
            1 for c in changed if c["from_mode"] == 0 and c["to_mode"] != 0
        ),
        "deactivated": sum(
            1 for c in changed if c["from_mode"] != 0 and c["to_mode"] == 0
        ),
        "blob_bytes": selector_blob_length(active),
        "shipped_blob_bytes": selector_blob_length(shipped_active),
        "delta_bytes": selector_blob_length(active)
        - selector_blob_length(shipped_active),
        "total_d_pose_gain": gain,
        "n600_mean_d_pose_gain": gain / FRAME_COUNT,
        "choices": choices,
    }


def price_adoption(adoption: dict[str, Any], *, base_d_pose: float) -> dict[str, Any]:
    """Compose the adoption's pose and rate legs against a declared base d_pose."""
    delta_bytes = int(adoption["delta_bytes"])
    new_d_pose = base_d_pose - float(adoption["n600_mean_d_pose_gain"])
    if new_d_pose <= 0.0:
        raise Fs1Error("projected d_pose is non-positive; the composition is invalid")
    ds_pose = pose_leg(new_d_pose) - pose_leg(base_d_pose)
    ds_rate = 25.0 * delta_bytes / UNCOMPRESSED_SOURCE_BYTES
    return {
        "base_d_pose": base_d_pose,
        "projected_d_pose": new_d_pose,
        "base_pose_leg": pose_leg(base_d_pose),
        "projected_pose_leg": pose_leg(new_d_pose),
        "delta_S_pose": ds_pose,
        "delta_bytes": delta_bytes,
        "delta_S_rate": ds_rate,
        "net_delta_S": ds_pose + ds_rate,
    }


def run_select(args) -> int:
    body = ShippedBody(Path(args.archive), Path(args.runtime))
    if args.require_frontier and body.archive_sha256 != FRONTIER_ARCHIVE_SHA256:
        raise Fs1Error(
            f"archive sha256 {body.archive_sha256} != frontier "
            f"{FRONTIER_ARCHIVE_SHA256}; refusing to select against an unidentified body"
        )
    sweep = json.loads(Path(args.sweep).read_text(encoding="utf-8"))
    if sweep.get("instrument", {}).get("archive_sha256") != body.archive_sha256:
        raise Fs1Error(
            "the sweep was measured on a different archive than the one being spliced"
        )
    report: dict[str, Any] = {
        "schema": "tac.ddm_fs1.select.v1",
        "axis": AXIS_ADVISORY,
        "score_claim": False,
        "promotable": False,
        "label": args.label,
        "shipped_body": body.facts(),
        "sweep": {
            "path": str(Path(args.sweep).resolve()),
            "sha256": sha256_file(Path(args.sweep)),
            "batch_size": sweep.get("batch_size"),
            "pairs_swept": len(sweep.get("pairs_swept", [])),
        },
        "base_d_pose_source": args.base_d_pose_source,
        "base_d_pose": args.base_d_pose,
        "variants": [],
    }
    emitted: dict[str, str] = dict(args.emit or [])
    report["emitted_choices"] = []
    for strategy in args.strategies:
        for margin in args.margins:
            adoption = adoption_from_sweep(
                sweep,
                margin=margin,
                shipped_choices=body.selector_choices,
                strategy=strategy,
                base_d_pose=args.base_d_pose,
            )
            priced = price_adoption(adoption, base_d_pose=args.base_d_pose)
            choices = adoption.pop("choices")
            variant = {
                **adoption,
                "price": priced,
                "projected_archive_bytes": body.archive_size
                + int(adoption["delta_bytes"]),
                "choices_sha256": sha256_array(choices),
            }
            report["variants"].append(variant)
            key = f"{strategy}@{margin:g}"
            if key in emitted:
                out = Path(emitted[key])
                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, choices)
                report["emitted_choices"].append(
                    {
                        "key": key,
                        "path": str(out),
                        "sha256": sha256_array(choices),
                        "bytes": out.stat().st_size,
                        "active_pairs": variant["active_pairs"],
                        "delta_bytes": variant["delta_bytes"],
                        "net_delta_S": priced["net_delta_S"],
                    }
                )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "label": args.label,
                "variants": [
                    {
                        "strategy": v["strategy"],
                        "margin": v["margin_gate"],
                        "changed": v["changed_count"],
                        "active": v["active_pairs"],
                        "blob": v["blob_bytes"],
                        "dB": v["delta_bytes"],
                        "min_ratio": v["min_adopted_ratio"],
                        "net_dS": v["price"]["net_delta_S"],
                    }
                    for v in report["variants"]
                ],
                "emitted": report["emitted_choices"],
            },
            indent=2,
        )
    )
    return 0


# --------------------------------------------------------------------------
# mode=build -- splice the archive and prove it.
# --------------------------------------------------------------------------


def run_build(args) -> int:
    body = ShippedBody(Path(args.archive), Path(args.runtime))
    if args.require_frontier and body.archive_sha256 != FRONTIER_ARCHIVE_SHA256:
        raise Fs1Error(
            f"archive sha256 {body.archive_sha256} != frontier {FRONTIER_ARCHIVE_SHA256}"
        )
    identity = control_identity(body)
    if not identity["identical_to_shipped_archive"]:
        raise Fs1Error(
            "the container identity control FAILED: rebuilding the shipped body with "
            "the unchanged tail does not reproduce the shipped bytes, so no byte delta "
            "this module reports would be the selector's. Refusing to build."
        )

    choices = np.load(Path(args.choices)).astype(np.uint8)
    if choices.shape != (FRAME_COUNT,):
        raise Fs1Error(f"choices must be ({FRAME_COUNT},), got {choices.shape}")
    if np.array_equal(choices, body.selector_choices):
        raise Fs1Error(
            "the requested selector equals the shipped one; there is nothing to build"
        )
    blob = encode_selector(choices)
    tail = stored_tail(blob)
    candidate = write_archive(
        body, tail, quality=BROTLI_QUALITY, lgwin=BROTLI_LGWIN
    )

    parsed = parse_back_parts(candidate, body.runtime_dir)
    base_parsed = parse_back_parts(body.archive_bytes_raw, body.runtime_dir)
    parsed_choices = parsed.pop("selector_choices")
    base_choices = base_parsed.pop("selector_choices")
    if not np.array_equal(parsed_choices, choices):
        differing = int(np.count_nonzero(parsed_choices != choices))
        raise Fs1Error(
            f"the written archive parses back to {differing} differing selector "
            "choices; refusing to return unverified bytes"
        )
    if not np.array_equal(base_choices, body.selector_choices):
        raise Fs1Error("the base parse-back disagrees with the shipped selector")
    unchanged_sections = {
        key: (parsed[key] == base_parsed[key])
        for key in parsed
        if key not in ("selector_blob_sha256", "selector_blob_bytes")
    }
    if not all(unchanged_sections.values()):
        broken = [k for k, v in unchanged_sections.items() if not v]
        raise Fs1Error(
            f"sections other than the selector changed: {broken}; the splice is not "
            "one-variable and refuses"
        )

    changed_pairs = np.flatnonzero(parsed_choices != base_choices).tolist()
    container_search = []
    if args.container_search:
        for quality, lgwin in CONTAINER_OPTIONS:
            alt = write_archive(body, tail, quality=quality, lgwin=lgwin)
            container_search.append(
                {
                    "brotli_quality": quality,
                    "brotli_lgwin": lgwin,
                    "archive_bytes": len(alt),
                    "delta_vs_shipped_shape": len(alt) - len(candidate),
                }
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_out = out_dir / "archive.zip"
    tmp = archive_out.with_suffix(".zip.partial")
    tmp.write_bytes(candidate)
    os.replace(tmp, archive_out)
    (out_dir / "selector_blob.bin").write_bytes(blob)
    np.save(out_dir / "selector_choices.npy", choices)
    (out_dir / "shipped_selector_blob.bin").write_bytes(body.selector_blob)

    report = {
        "schema": "tac.ddm_fs1.build.v1",
        "axis": "[bytes -- exact, device-free]",
        "score_claim": False,
        "promotable": False,
        "label": args.label,
        "shipped_body": body.facts(),
        "container_identity_control": identity,
        "candidate": {
            "path": str(archive_out),
            "sha256": sha256_bytes(candidate),
            "bytes": len(candidate),
            "delta_bytes": len(candidate) - body.archive_size,
            "selector_blob_bytes": len(blob),
            "selector_blob_sha256": sha256_bytes(blob),
            "selector_blob_delta_bytes": len(blob) - len(body.selector_blob),
            "selector_active_pairs": int(np.count_nonzero(choices)),
            "selector_active_index": np.flatnonzero(choices).tolist(),
            "selector_active_modes": choices[np.flatnonzero(choices)].tolist(),
            "stored_tail_bytes": len(tail),
            "stored_tail_delta_bytes": len(tail) - len(body.stored_tail_bytes),
        },
        "no_op_detector": {
            "changed_pairs": changed_pairs,
            "changed_pair_count": len(changed_pairs),
            "unchanged_pair_count": FRAME_COUNT - len(changed_pairs),
            "sections_byte_identical": unchanged_sections,
            "selector_blob_differs": parsed["selector_blob_sha256"]
            != base_parsed["selector_blob_sha256"],
        },
        "parse_back_candidate": parsed,
        "parse_back_base": base_parsed,
        "container_search": container_search,
        "rate_leg": {
            "base_rate_leg": 25.0 * body.archive_size / UNCOMPRESSED_SOURCE_BYTES,
            "candidate_rate_leg": 25.0 * len(candidate) / UNCOMPRESSED_SOURCE_BYTES,
            "delta_S_rate": 25.0
            * (len(candidate) - body.archive_size)
            / UNCOMPRESSED_SOURCE_BYTES,
        },
        "retained": {
            "archive": {"path": str(archive_out), "sha256": sha256_bytes(candidate)},
            "selector_blob": {
                "path": str(out_dir / "selector_blob.bin"),
                "sha256": sha256_bytes(blob),
            },
            "selector_choices": {
                "path": str(out_dir / "selector_choices.npy"),
                "sha256": sha256_array(choices),
            },
            "shipped_selector_blob": {
                "path": str(out_dir / "shipped_selector_blob.bin"),
                "sha256": sha256_bytes(body.selector_blob),
            },
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_sha256": report["candidate"]["sha256"],
                "candidate_bytes": report["candidate"]["bytes"],
                "delta_bytes": report["candidate"]["delta_bytes"],
                "selector_blob_bytes": report["candidate"]["selector_blob_bytes"],
                "changed_pairs": len(changed_pairs),
                "identity_control": identity["identical_to_shipped_archive"],
            },
            indent=2,
        )
    )
    return 0


# --------------------------------------------------------------------------
# mode=stage -- assemble the runtime tree MAIN fires.
# --------------------------------------------------------------------------


#: Filesystem metadata, never runtime content.  macOS writes an AppleDouble ``._x``
#: companion for every ``x`` on a volume without native fork support (the exFAT SSD
#: tier), so a tree copied there and compared byte-for-byte against an APFS source
#: would report dozens of phantom differences.  They are excluded from BOTH sides of
#: the comparison and from the staged tree, so the "only archive.zip may change"
#: guard keeps its meaning instead of being widened to pass.
_STAGE_IGNORE = ("__pycache__", "._*", ".DS_Store")


def _receiver_pin_only_diff(source: Path, target: Path) -> dict[str, Any]:
    """Prove the staged receiver differs from the source in the pin constants ONLY.

    A re-pinned ``inflate.py`` is the one legal receiver edit in this splice.  Trusting
    the helper to have touched nothing else would be a comment-only contract, so the
    two files are compared line by line and every differing line must assign
    ``ARCHIVE_SHA256`` or ``ARCHIVE_BYTES``.
    """
    from tac.candidate_seal import PIN_BYTES_NAME, PIN_SHA_NAME

    before = source.read_text(encoding="utf-8").splitlines()
    after = target.read_text(encoding="utf-8").splitlines()
    if len(before) != len(after):
        raise Fs1Error(
            f"the staged receiver changed line count ({len(before)} -> {len(after)}); a "
            "re-pin rewrites two lines in place and nothing else"
        )
    changed = [
        {"line": index + 1, "before": before[index], "after": after[index]}
        for index in range(len(before))
        if before[index] != after[index]
    ]
    offending = [
        row
        for row in changed
        if not row["after"].lstrip().startswith((PIN_SHA_NAME, PIN_BYTES_NAME))
    ]
    if offending:
        raise Fs1Error(
            f"the staged receiver differs outside its pin constants: {offending}"
        )
    if len(changed) != 2:
        raise Fs1Error(
            f"expected exactly the two pin lines to change, got {len(changed)}"
        )
    return {"changed_lines": changed, "pin_constants_only": True}


def _is_stage_noise(relative: Path) -> bool:
    return any(
        part == "__pycache__" or part == ".DS_Store" or part.startswith("._")
        for part in relative.parts
    )


def run_stage(args) -> int:
    source = Path(args.source_runtime).resolve()
    target = Path(args.out_dir).resolve()
    candidate = Path(args.archive).resolve()
    if target.exists() and not args.force:
        raise Fs1Error(f"refusing to overwrite an existing runtime tree: {target}")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        source, target, symlinks=False, ignore=shutil.ignore_patterns(*_STAGE_IGNORE)
    )
    shutil.copyfile(candidate, target / "archive.zip")
    # The public entrypoint PINS the archive it will accept
    # (``inflate.py:18-19`` ARCHIVE_SHA256 / ARCHIVE_BYTES, enforced at :28-31), so a
    # candidate archive cannot be staged under the shipped receiver unchanged: the
    # receiver would refuse its own tree.  Re-pin through the canonical helper, which
    # rewrites ONLY those two constant lines and restores the original bytes if the
    # result is not CONSISTENT.  ``_receiver_pin_only_diff`` below then PROVES the diff
    # is those two lines and nothing else, so "the receiver changed" stays a mechanical
    # identity update rather than a decode-behaviour change nobody audited.
    from tac.candidate_seal import repin_receiver

    repin = repin_receiver(target, archive_path=target / "archive.zip").to_dict()
    if repin["verdict_after"] != "CONSISTENT":
        raise Fs1Error(f"receiver re-pin did not reach CONSISTENT: {repin}")
    files = []
    for path in sorted(target.rglob("*")):
        if path.is_file() and not _is_stage_noise(path.relative_to(target)):
            files.append(
                {
                    "relative_path": str(path.relative_to(target)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    source_files = {
        str(p.relative_to(source)): sha256_file(p)
        for p in sorted(source.rglob("*"))
        if p.is_file() and not _is_stage_noise(p.relative_to(source))
    }
    target_files = {f["relative_path"]: f["sha256"] for f in files}
    differing = sorted(
        name
        for name in set(source_files) | set(target_files)
        if source_files.get(name) != target_files.get(name)
    )
    if differing != ["archive.zip", "inflate.py"]:
        raise Fs1Error(
            f"the staged runtime differs from the pointer tree in {differing}; only "
            "archive.zip and inflate.py's two pin constants may change"
        )
    receiver_diff = _receiver_pin_only_diff(
        source / "inflate.py", target / "inflate.py"
    )
    report = {
        "schema": "tac.ddm_fs1.stage.v1",
        "source_runtime": str(source),
        "staged_runtime": str(target),
        "archive_sha256": sha256_file(target / "archive.zip"),
        "archive_bytes": (target / "archive.zip").stat().st_size,
        "file_count": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files_differing_from_source": differing,
        "receiver_repin": repin,
        "receiver_diff_is_pin_constants_only": receiver_diff,
        "files": files,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("staged_runtime", "archive_sha256", "archive_bytes",
                       "file_count", "files_differing_from_source")}, indent=2))
    return 0


# --------------------------------------------------------------------------
# mode=measure -- batch-8 n600 d_pose through the CANDIDATE archive's own selector.
# --------------------------------------------------------------------------


def run_measure(args) -> int:
    # ``measure`` IS the confirm.  Its output is the number that gets priced, so
    # a screening backend has nothing to screen here -- it would only turn the
    # authority row into a screened row.  The flag exists so the receipt STATES
    # the backend instead of implying it, and so a caller cannot hand this path
    # an fp16 device by habit.  See ``tac.ane_screening`` (ddm_ane1).
    from tac.ane_screening import (
        AUTHORITY_BACKEND,
        AneScreeningError,
        assert_backend_name,
        backend_is_authority,
    )

    backend_name = assert_backend_name(getattr(args, "scorer_backend", AUTHORITY_BACKEND))
    if not backend_is_authority(backend_name):
        raise AneScreeningError(
            f"fs1 measure emits the priced d_pose, so it runs on "
            f"{AUTHORITY_BACKEND} fp32 only; {backend_name!r} is a SCREENING "
            "backend. Screen the mode choice in `pr1 selector --scorer-backend`, "
            "then confirm here."
        )

    import ddm_pr1_pose_resolve_on_renderer_change as pr1
    import ddm_up2_shipping_pose_solve as up2

    runtime = Path(args.runtime).resolve()
    observed = sha256_file(runtime / "archive.zip")
    # Gate BEFORE the instrument is built.  ``pr1.build_instrument``'s own sha gate is
    # pinned to the frontier body and would refuse a candidate, so this path passes it
    # the observed digest -- which makes THAT check vacuous.  The real identity gate is
    # this one, and it has to fire before six minutes of measurement, not after.
    if args.expect_archive_sha256 and observed != args.expect_archive_sha256:
        raise Fs1Error(
            f"runtime archive sha256 {observed} != expected {args.expect_archive_sha256}; "
            "refusing to measure an unidentified body"
        )
    instrument, meta = pr1.build_instrument(
        runtime=runtime,
        gt_cache=Path(args.gt_cache),
        axis=args.axis,
        renderer_source=Path(args.renderer),
        tokens_path=Path(args.tokens),
        archive_sha256=observed,
    )
    state = instrument.state
    codes = state.codes.copy()
    coefficients = up2.codes_to_coefficients(codes, state.coefficient_scales)
    pairs = np.arange(FRAME_COUNT, dtype=np.int64)
    started = time.time()
    per_pair, poses = up2.measure_pose(
        instrument.posenet,
        state,
        coefficients,
        instrument.raw,
        instrument.targets,
        pairs,
        batch_size=args.batch_size,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload_dir = out.parent / f"{out.stem}_payload"
    payload_dir.mkdir(parents=True, exist_ok=True)
    per_pair_path = payload_dir / "per_pair_d_pose.npy"
    poses_path = payload_dir / "pose_vectors.npy"
    choices_path = payload_dir / "selector_choices.npy"
    np.save(per_pair_path, per_pair)
    np.save(poses_path, poses)
    np.save(choices_path, state.selector_choices)
    report = {
        "schema": "tac.ddm_fs1.measure.v1",
        "axis": AXIS_ADVISORY,
        "score_claim": False,
        "promotable": False,
        "label": args.label,
        "instrument": meta,
        "measured_archive_sha256": observed,
        "scorer_backend": backend_name,
        "authority_backend": AUTHORITY_BACKEND,
        "batch_size": args.batch_size,
        "pairs": int(pairs.size),
        "pair_selection": "full n600",
        "codes_source": "the archive's own carrier codes",
        "codes_sha256": sha256_array(codes),
        "selector_active_pairs": np.flatnonzero(state.selector_choices).tolist(),
        "selector_choices_sha256": sha256_array(state.selector_choices),
        "d_pose_mean": float(per_pair.mean()),
        "d_pose_median": float(np.median(per_pair)),
        "pose_leg": pose_leg(float(per_pair.mean())),
        "elapsed_seconds": time.time() - started,
        "payload": {
            "per_pair_d_pose": {
                "path": str(per_pair_path),
                "sha256": sha256_array(per_pair),
                "bytes": per_pair_path.stat().st_size,
            },
            "pose_vectors": {
                "path": str(poses_path),
                "sha256": sha256_array(poses),
                "bytes": poses_path.stat().st_size,
            },
            "selector_choices": {
                "path": str(choices_path),
                "sha256": sha256_array(state.selector_choices),
                "bytes": choices_path.stat().st_size,
            },
        },
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("label", "pairs", "d_pose_mean", "pose_leg",
                       "elapsed_seconds")}, indent=2))
    return 0


# --------------------------------------------------------------------------
# mode=compose -- the closing arithmetic, base vs candidate, one instrument.
# --------------------------------------------------------------------------


def _bootstrap_admissibility(
    base_per_pair: np.ndarray,
    candidate_per_pair: np.ndarray,
    *,
    delta_bytes: int,
    exact_net: float,
) -> dict[str, Any]:
    """Apply the registered near-win acceptance rule, not a hand-rolled tolerance.

    ``tac.canonical_equations.exchange_ratio_noise_floor_v1`` states the admission
    test for a byte<->distortion near win: resample the 600 PAIRS with replacement
    (site-level resampling is forbidden -- ``ddm_fs3`` measured AVERAGE != MARGINAL
    by 2.24x) and ADMIT iff ``quantile_0.975(dS_b) < 0``.  A point estimate below
    zero is not sufficient.

    Two features of THIS edit make the test unusually clean, and both are stated
    rather than assumed:

    * ``dB`` is a whole-archive constant (a selector blob is not a per-pair byte
      stream), so it enters the law's fixed calibration term ``c`` and carries no
      resample dispersion of its own -- consistent with the law's own measured
      ``sigma_B = 0``;
    * ``d_seg`` is unchanged by construction (the selector writes frame 2p, SegNet
      reads frame 2p+1), so the seg leg contributes exactly zero to every resample.

    The SAME draw matrix drives both pose vectors, or the pairing between the base
    and the candidate is destroyed and the interval is not an interval of a delta.
    """
    from tac.canonical_equations.exchange_ratio_noise_floor_20260903 import (
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
        bootstrap_mean,
        delta_s_from_components,
        draw_pair_indices,
        near_win_is_admissible,
        percentile_interval_95,
    )

    draws = draw_pair_indices()
    base_mean = float(np.asarray(base_per_pair, dtype=np.float64).mean())
    cand_mean = float(np.asarray(candidate_per_pair, dtype=np.float64).mean())
    base_samples = bootstrap_mean(base_per_pair, draws, exact_mean=base_mean)
    cand_samples = bootstrap_mean(candidate_per_pair, draws, exact_mean=cand_mean)
    delta_s = delta_s_from_components(
        base_d_seg=0.0,
        candidate_d_seg=0.0,
        base_d_pose=base_samples,
        candidate_d_pose=cand_samples,
        delta_bytes=float(delta_bytes),
    )
    low, high = percentile_interval_95(delta_s)
    return {
        "law": "tac.canonical_equations exchange_ratio_noise_floor_v1",
        "rule": "ADMIT iff quantile_0.975(dS_b) < 0 over a seeded n600 PAIR bootstrap",
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "point_net_delta_S": exact_net,
        "interval_95_low": low,
        "interval_95_high": high,
        "half_width": 0.5 * (high - low),
        "admissible": bool(near_win_is_admissible(delta_s)),
        "delta_bytes_is_a_whole_archive_constant": True,
        "delta_d_seg_is_zero_by_construction": True,
    }


def run_compose(args) -> int:
    base = json.loads(Path(args.base_measure).read_text(encoding="utf-8"))
    cand = json.loads(Path(args.candidate_measure).read_text(encoding="utf-8"))
    build = json.loads(Path(args.build_report).read_text(encoding="utf-8"))
    for key in ("gt_cache", "renderer"):
        if base["instrument"][key] != cand["instrument"][key]:
            raise Fs1Error(
                f"base and candidate measures disagree on instrument.{key}; a delta "
                "across two instruments is not a delta"
            )
    if base["batch_size"] != cand["batch_size"]:
        raise Fs1Error("base and candidate were measured at different batch shapes")
    if base["codes_sha256"] != cand["codes_sha256"]:
        raise Fs1Error("base and candidate carry different carrier codes")
    if base["selector_choices_sha256"] == cand["selector_choices_sha256"]:
        raise Fs1Error(
            "base and candidate carry the SAME selector; this would report a null "
            "difference as a result"
        )
    if base["measured_archive_sha256"] != build["shipped_body"]["archive_sha256"]:
        raise Fs1Error("the base measure was not taken on the body the build spliced")
    if cand["measured_archive_sha256"] != build["candidate"]["sha256"]:
        raise Fs1Error("the candidate measure was not taken on the built archive")

    base_pp = np.load(base["payload"]["per_pair_d_pose"]["path"])
    cand_pp = np.load(cand["payload"]["per_pair_d_pose"]["path"])
    base_d_pose = float(base_pp.mean())
    cand_d_pose = float(cand_pp.mean())
    base_bytes = int(build["shipped_body"]["archive_bytes"])
    cand_bytes = int(build["candidate"]["bytes"])
    delta_bytes = cand_bytes - base_bytes

    ds_pose = pose_leg(cand_d_pose) - pose_leg(base_d_pose)
    ds_rate = 25.0 * delta_bytes / UNCOMPRESSED_SOURCE_BYTES
    net = ds_pose + ds_rate

    changed = [int(p) for p in build["no_op_detector"]["changed_pairs"]]
    if not changed:
        raise Fs1Error("the build report records no changed pair; nothing to compose")
    unchanged_mask = np.ones(FRAME_COUNT, dtype=bool)
    unchanged_mask[np.asarray(changed, dtype=np.int64)] = False
    if not unchanged_mask.any():
        raise Fs1Error("every pair changed; the unchanged-pair control would be vacuous")

    per_pair_rows = [
        {
            "pair": int(p),
            "base_d_pose": float(base_pp[p]),
            "candidate_d_pose": float(cand_pp[p]),
            "gain": float(base_pp[p] - cand_pp[p]),
            "ratio": (
                float(base_pp[p] / cand_pp[p]) if cand_pp[p] > 0 else float("inf")
            ),
        }
        for p in changed
    ]
    unchanged_max_abs = float(np.abs(base_pp[unchanged_mask] - cand_pp[unchanged_mask]).max())

    projected_from_sweep = float(args.projected_net_dS) if args.projected_net_dS else None
    reproduction = None
    if projected_from_sweep is not None:
        reproduction = {
            "sweep_projection_net_dS": projected_from_sweep,
            "measured_net_dS": net,
            "relative_difference": (net - projected_from_sweep)
            / abs(projected_from_sweep),
            "within_exchange_noise_floor_6pct": bool(
                abs(net - projected_from_sweep) / abs(projected_from_sweep) <= 0.06
            ),
        }

    admissibility = _bootstrap_admissibility(
        base_pp, cand_pp, delta_bytes=delta_bytes, exact_net=net
    )

    report = {
        "schema": "tac.ddm_fs1.compose.v1",
        "axis": AXIS_ADVISORY,
        "score_claim": False,
        "promotable": False,
        "label": args.label,
        "instrument": base["instrument"],
        "batch_size": base["batch_size"],
        "base": {
            "archive_sha256": build["shipped_body"]["archive_sha256"],
            "archive_bytes": base_bytes,
            "d_pose": base_d_pose,
            "pose_leg": pose_leg(base_d_pose),
            "measure": str(Path(args.base_measure).resolve()),
        },
        "candidate": {
            "archive_sha256": build["candidate"]["sha256"],
            "archive_bytes": cand_bytes,
            "d_pose": cand_d_pose,
            "pose_leg": pose_leg(cand_d_pose),
            "measure": str(Path(args.candidate_measure).resolve()),
        },
        "delta": {
            "d_pose": cand_d_pose - base_d_pose,
            "delta_S_pose": ds_pose,
            "delta_bytes": delta_bytes,
            "delta_S_rate": ds_rate,
            "net_delta_S": net,
            "delta_S_seg": 0.0,
            "delta_S_seg_justification": (
                "STRUCTURAL, verified at source: the selector writes only "
                "output[2*frame_ids] (f26_inflate.py:133), i.e. frame 2p of each pair, "
                "while SegNet scores x[:, -1, ...] (upstream/modules.py:100), i.e. "
                "frame 2p+1. The build report's no-op detector additionally proves the "
                "semantic section, token stream, HPAC model, residual table and CAP1 "
                "carrier are byte-identical, so the odd frames are bit-identical and "
                "d_seg cannot move."
            ),
        },
        "per_pair_changed": per_pair_rows,
        "unchanged_pairs": {
            "count": int(unchanged_mask.sum()),
            "max_abs_d_pose_difference": unchanged_max_abs,
            "bit_identical": bool(unchanged_max_abs == 0.0),
        },
        "projection_reproduction": reproduction,
        "admissibility": admissibility,
        "t4_composition": {
            "afr1_score_T4": AFR1_SCORE_T4,
            "afr1_d_seg_T4": AFR1_D_SEG_T4,
            "afr1_d_pose_T4": AFR1_D_POSE_T4,
            "afr1_recomputed_from_components": composed_score(
                AFR1_D_SEG_T4, AFR1_D_POSE_T4, FRONTIER_ARCHIVE_BYTES
            ),
            "projected_candidate_S": AFR1_SCORE_T4 + net,
            "composition_note": (
                "the LEVEL is the contest-CUDA T4 receipt; the DELTA is this arm's "
                "advisory same-instrument difference. Labelled "
                "[macOS-CPU advisory projection] and NOT a score."
            ),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "base_d_pose": base_d_pose,
                "candidate_d_pose": cand_d_pose,
                "delta_S_pose": ds_pose,
                "delta_bytes": delta_bytes,
                "delta_S_rate": ds_rate,
                "net_delta_S": net,
                "projected_candidate_S": AFR1_SCORE_T4 + net,
                "reproduction": reproduction,
                "admissibility": admissibility,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # A net dS is a small NEGATIVE number in scientific notation.  argparse's built-in
    # negative-number matcher understands "-1" and "-1.5" but not "-1.03e-04", so it
    # classifies the value as an unknown option and dies with "expected one argument".
    # Same widening, for the same reason, as tools/make_candidate_seal.py.
    parser._negative_number_matcher = re.compile(
        r"^-\d+$|^-\d*\.\d+$|^-\d*\.?\d+[eE][+-]?\d+$"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser("select", help="adopt pr1's sweep under a margin gate and price it")
    select.add_argument("--archive", required=True)
    select.add_argument("--runtime", required=True)
    select.add_argument("--sweep", required=True)
    select.add_argument("--margins", type=float, nargs="+", default=[1.01, 1.0])
    select.add_argument(
        "--strategies",
        nargs="+",
        default=["ratio_gate", "byte_optimal"],
        choices=["ratio_gate", "byte_optimal"],
    )
    select.add_argument("--base-d-pose", type=float, required=True)
    select.add_argument("--base-d-pose-source", required=True)
    select.add_argument(
        "--emit",
        action="append",
        nargs=2,
        metavar=("STRATEGY@MARGIN", "PATH"),
        default=None,
        help="write one variant's choices to PATH, e.g. --emit byte_optimal@1.01 out.npy",
    )
    select.add_argument("--require-frontier", action="store_true", default=True)
    select.add_argument("--label", default="select")
    select.add_argument("--out", required=True)
    select.set_defaults(func=run_select)

    build = sub.add_parser("build", help="splice the candidate archive and prove it")
    build.add_argument("--archive", required=True)
    build.add_argument("--runtime", required=True)
    build.add_argument("--choices", required=True)
    build.add_argument("--out-dir", required=True)
    build.add_argument("--container-search", action="store_true")
    build.add_argument("--require-frontier", action="store_true", default=True)
    build.add_argument("--label", default="build")
    build.add_argument("--out", required=True)
    build.set_defaults(func=run_build)

    stage = sub.add_parser("stage", help="assemble the runtime tree MAIN fires")
    stage.add_argument("--source-runtime", required=True)
    stage.add_argument("--archive", required=True)
    stage.add_argument("--out-dir", required=True)
    stage.add_argument("--force", action="store_true")
    stage.add_argument("--out", required=True)
    stage.set_defaults(func=run_stage)

    measure = sub.add_parser("measure", help="batch-8 n600 d_pose through an archive's own selector")
    measure.add_argument("--runtime", required=True)
    measure.add_argument("--gt-cache", required=True)
    measure.add_argument("--renderer", required=True)
    measure.add_argument("--tokens", required=True)
    measure.add_argument("--axis", default="contest_cuda")
    measure.add_argument("--batch-size", type=int, default=8)
    measure.add_argument("--threads", type=int, default=4)
    measure.add_argument("--expect-archive-sha256", default=None)
    measure.add_argument(
        "--scorer-backend", default="cpu_torch",
        choices=("cpu_torch", "coreml_cpu_fp32", "ane_fp16_screen"),
        help=(
            "recorded in the receipt; this path emits the PRICED d_pose so it "
            "refuses every non-authority backend (tac.ane_screening)."
        ),
    )
    measure.add_argument("--label", default="measure")
    measure.add_argument("--out", required=True)
    measure.set_defaults(func=run_measure)

    compose = sub.add_parser("compose", help="the closing arithmetic, one instrument")
    compose.add_argument("--base-measure", required=True)
    compose.add_argument("--candidate-measure", required=True)
    compose.add_argument("--build-report", required=True)
    compose.add_argument("--projected-net-dS", type=float, default=None)
    compose.add_argument("--label", default="compose")
    compose.add_argument("--out", required=True)
    compose.set_defaults(func=run_compose)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    threads = int(getattr(args, "threads", 0) or 0)
    if threads:
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        try:
            import torch

            torch.set_num_threads(threads)
        except ImportError:
            pass
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
