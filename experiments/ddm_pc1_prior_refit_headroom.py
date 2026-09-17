#!/usr/bin/env python3
"""ddm_pc1 -- is a HPAC prior REFIT owed at the live pointer?  Answer it in seconds.

WHY THIS EXISTS
---------------
The prior is the token tail's coder model.  It is fit to a token field; every field move
leaves it a little stale, and a refit converts staleness into bytes.  The lab has paid for
that conversion twice and the two rows disagree by an order of magnitude:

* ``hpr1`` (move 47) refit a prior whose training field was **11,128 token sites** away from
  the live field and took **-887 B** (model +351 B, tail -1,238 B).
* ``dpi1`` (move 48) refit at **0 sites** of drift and took **+37 B** -- it LOST.

So the question "is a refit owed now?" is really "how far has the field drifted?", and that
is a numpy diff, not a five-hour Metal burn.  This module is that diff, with the two guards
that make its answer trustworthy.

THE TWO GUARDS (both earned, both cheap)
----------------------------------------
1. **Which field was the shipped prior actually fit on?**  Never inferred from a memo.  This
   module reads the prior SECTION out of the live archive and out of the candidate ancestor
   archives and reports byte identity.  The ancestor whose prior section is byte-identical to
   the live one is the run that produced it, and that run's training cache is the fit field.

2. **Is the file you handed me a TOKEN field at all?**  This is the confound this module was
   written for.  A token plane and a SegNet argmax plane have the SAME dtype and the SAME
   ``(600, 384, 512)`` shape, so a mis-named file substitutes silently -- and on this object
   it does so by a factor of 150 (``argmax_move54.npy`` differs from the move-54 token field
   at 27,303 sites, which reads as "very stale" when the true drift is 179).  Every field
   here must therefore arrive WITH a declared sha256 that this module verifies.  A field
   without a verified sha is refused, never measured.

WHAT IT RETURNS
---------------
The drifted-site count, the headroom implied by the measured calibration slope, and the
verdict against the caller's fire bar.  The slope is n=1 and is labelled DERIVED wherever it
is used: this ranks a refit rung, it does not price one.  A rung that clears the bar here
still has to be trained and priced by the real coder.

Axis ``[macOS-CPU advisory; exact counts over retained payloads]``; ``score_claim=false``.

Usage::

  python experiments/ddm_pc1_prior_refit_headroom.py --out <store>/HEADROOM.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np

#: The RX1M payload header: magic, version, codec, reserved, flags, and the three
#: stored section lengths.  Identical in ``submissions/mrs7/FORMAT.md`` and in the
#: shipped receiver; read here only to SPLIT the member, never to change it.
PAYLOAD_HEADER = struct.Struct("<4sBBBBHHH")
FIELD_SHAPE = (600, 384, 512)
FIELD_VALUES = FIELD_SHAPE[0] * FIELD_SHAPE[1] * FIELD_SHAPE[2]

#: MEASURED calibration, both rows byte-closed through the real coder on this object.
#: ``sites`` is the token-site distance between the prior's own training field and the
#: field it was asked to code; ``delta_bytes`` is the archive delta the refit realized.
#:
#: Provenance, stated because it is easy to misread: ``delta_bytes`` and ``legs`` are the
#: arms' own published rows.  ``sites`` is NOT -- neither arm published a drift count.
#: ddm_pc1 measured both counts itself (2026-09-17) from the retained training fields:
#: 11,128 = cl2's fit field ``cc10a7b0...`` against hpr1's ``a92e7d90...``; 0 = dpi1
#: refit on the very field the shipped prior was already fit to.
CALIBRATION = (
    {
        "row": "hpr1 retrain control -> pointer move 47",
        "sites": 11128,
        "delta_bytes": -887,
        "legs": {"model_bytes": +351, "tail_bytes": -1238},
        "memo": ".omx/research/ddm_hpr1_retrain_control_move46_contest_cuda_20260911_pointer_move_47_20260911.md",
    },
    {
        "row": "dpi1 depth-restored refit on move 48",
        "sites": 0,
        "delta_bytes": +37,
        "legs": {"model_bytes": -400, "tail_bytes": +437},
        "memo": ".omx/research/ddm_gs3_gestalt_after_submission_20260903.md (Addendum 49)",
    },
)


class HeadroomError(RuntimeError):
    """An input is not what the caller declared it to be."""


@dataclass(frozen=True)
class Field:
    """A token plane whose identity was verified, never assumed."""

    label: str
    path: Path
    sha256: str
    plane: np.ndarray


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def split_member(member: bytes) -> dict[str, bytes]:
    """Split an RX1M payload into its four sections without rewriting a byte."""
    if len(member) < PAYLOAD_HEADER.size:
        raise HeadroomError("payload is shorter than its own header")
    magic, _version, _codec, _reserved, _flags, prior_n, renderer_n, carrier_n = (
        PAYLOAD_HEADER.unpack_from(member)
    )
    if magic != b"RX1M":
        raise HeadroomError(f"payload magic is {magic!r}, not RX1M")
    offset = PAYLOAD_HEADER.size
    sections: dict[str, bytes] = {}
    for name, length in (("prior", prior_n), ("renderer", renderer_n), ("carrier", carrier_n)):
        sections[name] = member[offset : offset + length]
        if len(sections[name]) != length:
            raise HeadroomError(f"{name} section is truncated")
        offset += length
    sections["tail"] = member[offset:]
    return sections


def archive_sections(archive: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != 1:
            raise HeadroomError(f"{archive} holds {len(names)} members, not one")
        member = bundle.read(names[0])
    return split_member(member)


def prior_identity(live: Path, ancestors: dict[str, Path]) -> dict:
    """Name the run that produced the shipped prior, by byte identity of its section.

    Guard 1.  The shipped prior's training field is a property of the RUN that produced
    it.  An ancestor archive whose prior section is byte-identical to the live one IS that
    run's output, so its training cache is the fit field.  Anything weaker is a memo claim.
    """
    live_prior = archive_sections(live)["prior"]
    live_sha = sha256_bytes(live_prior)
    rows = {}
    for label, path in ancestors.items():
        if not path.is_file():
            rows[label] = {"present": False}
            continue
        prior = archive_sections(path)["prior"]
        rows[label] = {
            "present": True,
            "archive": str(path),
            "prior_bytes": len(prior),
            "prior_sha256": sha256_bytes(prior),
            "identical_to_live": sha256_bytes(prior) == live_sha,
        }
    matches = [label for label, row in rows.items() if row.get("identical_to_live")]
    return {
        "live_archive": str(live),
        "live_prior_bytes": len(live_prior),
        "live_prior_sha256": live_sha,
        "ancestors": rows,
        "produced_by": matches,
    }


def load_field(label: str, path: Path, expected_sha256: str) -> Field:
    """Load a token plane and REFUSE it unless its bytes are the ones declared.

    Guard 2.  A SegNet argmax plane has this dtype and this shape too.  The sha is the
    only thing that separates them, so it is required, not optional.
    """
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            missing = [str(i) for i in range(FIELD_SHAPE[0]) if str(i) not in data.files]
            if missing:
                raise HeadroomError(f"{path} is missing {len(missing)} planes")
            plane = np.stack([data[str(i)] for i in range(FIELD_SHAPE[0])])
    elif path.suffix == ".npy":
        plane = np.load(path, allow_pickle=False)
    else:
        plane = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        if plane.size != FIELD_VALUES:
            raise HeadroomError(f"{path} holds {plane.size} bytes, not {FIELD_VALUES}")
        plane = plane.reshape(FIELD_SHAPE)
    plane = np.ascontiguousarray(plane)
    if plane.shape != FIELD_SHAPE or plane.dtype != np.uint8:
        raise HeadroomError(f"{path} geometry is {plane.shape} {plane.dtype}, not uint8 {FIELD_SHAPE}")
    if int(plane.max()) > 4:
        raise HeadroomError(f"{path} carries symbol {int(plane.max())}, outside the 5-symbol alphabet")
    observed = sha256_bytes(plane.tobytes())
    if observed != expected_sha256:
        raise HeadroomError(
            f"{label}: declared sha256 {expected_sha256} but the plane hashes to {observed}; "
            "refusing -- a mis-named plane (a SegNet argmax, say) reads as staleness that is not there"
        )
    return Field(label=label, path=path, sha256=observed, plane=plane)


def calibration_slope() -> dict:
    """Bytes per drifted site, from the one row that converted drift into bytes."""
    positive = [row for row in CALIBRATION if row["sites"] > 0]
    if len(positive) != 1:
        raise HeadroomError("the slope is n=1 by construction; a second row changes this function")
    row = positive[0]
    intercepts = [entry["delta_bytes"] for entry in CALIBRATION if entry["sites"] == 0]
    return {
        "bytes_per_site": row["delta_bytes"] / row["sites"],
        "from_row": row["row"],
        "n": 1,
        "zero_drift_rows": intercepts,
        "label": "DERIVED (n=1 slope; ranks a rung, never prices one)",
    }


def headroom(fit: Field, live: Field, *, fire_bar_bytes: float) -> dict:
    sites = 0 if fit.sha256 == live.sha256 else int((fit.plane != live.plane).sum())
    slope = calibration_slope()
    predicted = sites * slope["bytes_per_site"]
    zero_drift = slope["zero_drift_rows"]
    with_intercept = predicted + (sum(zero_drift) / len(zero_drift) if zero_drift else 0.0)
    return {
        "fit_field": {"label": fit.label, "path": str(fit.path), "sha256": fit.sha256},
        "live_field": {"label": live.label, "path": str(live.path), "sha256": live.sha256},
        "drifted_sites": sites,
        "field_values": FIELD_VALUES,
        "drifted_fraction": sites / FIELD_VALUES,
        "calibration": slope,
        "predicted_delta_bytes_slope_only": predicted,
        "predicted_delta_bytes_with_zero_drift_intercept": with_intercept,
        "fire_bar_bytes": fire_bar_bytes,
        # The verdict reads the SLOPE-ONLY number on purpose.  That is the optimistic
        # estimate -- it spends the whole staleness credit and charges nothing for the
        # model leg the one zero-drift row measured at +37 B.  A refusal that survives the
        # optimistic estimate is the stronger refusal; an OWED here is a floor, not a promise.
        "verdict": "REFIT_OWED" if predicted <= fire_bar_bytes else "REFIT_NOT_OWED",
        "label": "DERIVED from a measured n=1 slope; a rung that clears the bar must still be trained and priced",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--live-archive", type=Path, required=True)
    parser.add_argument("--fit-field", type=Path, required=True)
    parser.add_argument("--fit-field-sha256", required=True)
    parser.add_argument("--live-field", type=Path, required=True)
    parser.add_argument("--live-field-sha256", required=True)
    parser.add_argument(
        "--ancestor",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="an ancestor archive whose prior section may be the shipped one",
    )
    parser.add_argument("--fire-bar-bytes", type=float, default=-25.0)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ancestors: dict[str, Path] = {}
    for entry in args.ancestor:
        label, _, path = entry.partition("=")
        if not label or not path:
            raise HeadroomError(f"--ancestor wants LABEL=PATH, got {entry!r}")
        ancestors[label] = Path(path)
    identity = prior_identity(args.live_archive, ancestors)
    fit = load_field("prior_training_field", args.fit_field, args.fit_field_sha256)
    live = load_field("live_token_field", args.live_field, args.live_field_sha256)
    result = {
        "schema": "ddm_pc1_prior_refit_headroom.v1",
        "axis": "[macOS-CPU advisory; exact counts over retained payloads]",
        "score_claim": False,
        "prior_identity": identity,
        "headroom": headroom(fit, live, fire_bar_bytes=args.fire_bar_bytes),
        "calibration_rows": list(CALIBRATION),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["headroom"], indent=2, sort_keys=True))
    print(json.dumps({"produced_by": identity["produced_by"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
