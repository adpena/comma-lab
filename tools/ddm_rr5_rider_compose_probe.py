#!/usr/bin/env python3
"""ddm_rr5 -- MEASURE whether the rider and jg2's token edits commute.

The harvest chain wants to apply BOTH the ra2+ra1 CPR1 rider and jg2's
token-edit payload to the same body.  Order of operations is a real question --
so it is ANSWERED BY MEASUREMENT here, not by an argument about disjointness.

The two orders are constructed on the real bytes:

  edits-then-rider  rider applied to jg2's retained edited candidate.
  rider-then-edits  jg2's edited TAIL spliced into the rider-applied pointer.

The second order needs no re-run of jg2's expensive encode: jg2's tail is a pure
function of (token field, edits) and never reads the carrier, so its retained
candidate's tail IS the tail that order would produce.  That claim is not
assumed either -- the probe first verifies that jg2's candidate differs from the
pointer ONLY in the tail section, and refuses otherwise.

Both orders are then compared byte-for-byte, and each is re-parsed through the
REAL receiver so a commuting pair cannot be a pair of equally-broken archives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from ddm_rr5_rider_apply import (  # noqa: E402
    S_PER_BYTE,
    RiderApplyError,
    _emit_zip,
    _load_receiver,
    _sha256,
    apply_rider,
    parse_container,
)


def _sections(container) -> dict[str, bytes]:
    return {
        "hpac": container.hpac_stream,
        "semantic": container.semantic_stream,
        "carrier": container.carrier_stream,
        "tail": container.section_tail,
    }


def _splice_tail(container, tail: bytes) -> bytes:
    """Replace only the section tail; every other byte is carried through."""
    ra = container.residual_archive
    outer = b"".join(
        (
            ra.RX1_MODEL_HEADER.pack(
                container.magic,
                container.version,
                container.codec,
                container.table_mode,
                container.reserved,
                len(container.hpac_stream),
                len(container.semantic_stream),
                len(container.carrier_stream),
            ),
            container.hpac_stream,
            container.semantic_stream,
            container.carrier_stream,
            tail,
        )
    )
    return _emit_zip(container, outer)


def run(
    pointer: Path,
    runtime: Path,
    edited: Path,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    pointer_container = parse_container(pointer, runtime, expect_sha256=None)
    edited_container = parse_container(edited, runtime, expect_sha256=None)

    left, right = _sections(pointer_container), _sections(edited_container)
    differing = sorted(name for name in left if left[name] != right[name])
    if differing != ["tail"]:
        raise RiderApplyError(
            f"jg2's candidate differs from the pointer in {differing}, not only the "
            "tail; the rider-then-edits construction below would not be faithful"
        )

    # --- order A: edits, then rider --------------------------------------- #
    order_a = apply_rider(
        edited,
        runtime,
        out_dir / "order_a_edits_then_rider",
        expect_sha256=None,
        full=False,
    )
    order_a_bytes = Path(order_a["output"]["archive_path"]).read_bytes()

    # --- order B: rider, then edits --------------------------------------- #
    rider_only = apply_rider(
        pointer,
        runtime,
        out_dir / "rider_only",
        expect_sha256=None,
        full=False,
    )
    rider_archive = Path(rider_only["output"]["archive_path"])
    rider_container = parse_container(rider_archive, runtime, expect_sha256=None)
    order_b_bytes = _splice_tail(rider_container, edited_container.section_tail)
    order_b_path = out_dir / "order_b_rider_then_edits.zip"
    order_b_path.write_bytes(order_b_bytes)

    # --- both orders must still PARSE through the real receiver ----------- #
    rider_runtime = Path(rider_only["output"]["runtime_dir"])
    parses: dict[str, Any] = {}
    for label, path in (("order_a", Path(order_a["output"]["archive_path"])),
                        ("order_b", order_b_path)):
        ra = _load_receiver(rider_runtime)
        parts = ra.read_residual_archive(path)
        parses[label] = {
            "schema": parts.schema,
            "token_codec": parts.token_codec,
            "carrier_blob_sha256": _sha256(parts.carrier_blob),
            "token_stream_sha256": _sha256(parts.token_stream),
            "token_stream_bytes": len(parts.token_stream),
        }

    pointer_bytes = len(pointer_container.archive_bytes)
    edited_bytes = len(edited_container.archive_bytes)
    commutes = order_a_bytes == order_b_bytes

    report = {
        "arm": "ddm_rr5",
        "schema": "ddm_rr5_compose_probe.v1",
        "axis": "[byte-exact, lossless rider x lossless-parse token edits]",
        "score_claim": False,
        "promotable": False,
        "inputs": {
            "pointer": {"path": str(pointer), "bytes": pointer_bytes,
                        "sha256": _sha256(pointer_container.archive_bytes)},
            "edited": {"path": str(edited), "bytes": edited_bytes,
                       "sha256": _sha256(edited_container.archive_bytes)},
            "sections_differing_pointer_vs_edited": differing,
        },
        "order_a_edits_then_rider": {
            "archive_bytes": len(order_a_bytes),
            "archive_sha256": _sha256(order_a_bytes),
            "rider_delta_bytes": order_a["realized"]["archive_delta_bytes"],
        },
        "order_b_rider_then_edits": {
            "archive_bytes": len(order_b_bytes),
            "archive_sha256": _sha256(order_b_bytes),
            "path": str(order_b_path),
        },
        "rider_alone_on_pointer": {
            "archive_bytes": rider_only["output"]["archive_bytes"],
            "archive_sha256": rider_only["output"]["archive_sha256"],
            "delta_bytes": rider_only["realized"]["archive_delta_bytes"],
            "delta_S": rider_only["realized"]["delta_S"],
        },
        "verdict": {
            "orders_commute": commutes,
            "order_delta_bytes": len(order_a_bytes) - len(order_b_bytes),
            "correct_order_for_harvest": "either (MEASURED byte-identical)"
            if commutes
            else "edits first, then rider (the rider must see final carrier bytes)",
            "both_parse": parses,
        },
        "composed_arithmetic": {
            "edits_delta_bytes": pointer_bytes - edited_bytes,
            "rider_delta_bytes": rider_only["realized"]["archive_delta_bytes"],
            "composed_delta_bytes": pointer_bytes - len(order_a_bytes),
            "additive": (pointer_bytes - len(order_a_bytes))
            == (pointer_bytes - edited_bytes)
            + rider_only["realized"]["archive_delta_bytes"],
            "composed_delta_S": -(pointer_bytes - len(order_a_bytes)) * S_PER_BYTE,
        },
    }
    (out_dir / "RR5_COMPOSE_REPORT.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pointer",
        type=Path,
        default=Path("/Volumes/APDataStore/pact/ddm_up3/candidate_runtime/archive.zip"),
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path("/Volumes/APDataStore/pact/ddm_up3/candidate_runtime"),
    )
    parser.add_argument(
        "--edited",
        type=Path,
        default=Path(
            "/Volumes/APDataStore/pact/ddm_jg2/retained/candidate_jg1_3pair.zip"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Volumes/APDataStore/pact/ddm_rr5/retained/compose"),
    )
    args = parser.parse_args()
    report = run(
        args.pointer.resolve(),
        args.runtime.resolve(),
        args.edited.resolve(),
        args.out_dir.resolve(),
    )
    print(json.dumps(report["verdict"], indent=2, default=str))
    print(json.dumps(report["composed_arithmetic"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
