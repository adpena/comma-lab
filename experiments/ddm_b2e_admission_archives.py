#!/usr/bin/env python3
"""Splice the burn-2 semantic state and its three replayed edits into real archives.

WHY THIS FILE EXISTS
--------------------
``experiments/ddm_b2e_edit_replay_admission.py`` replays the three MP2 edit
constructions on a burn-2 checkpoint and adjudicates them against a
pre-registered collapse bar, but its ``admit`` stage needs MEASURED pose numbers.
Pose is only measurable through a real decode of real archive bytes.  This module
is the missing leg: it turns the burn-2 EMA state and each replayed edit into a
complete, receiver-closed ``archive.zip`` plus candidate-bound runtime, using the
SAME splice mechanics MP2 used for the calibration rows.

WHAT IS REUSED, NEVER REIMPLEMENTED
-----------------------------------
* the edit constructions -- ``b2e.build_mixed_q3q4`` / ``b2e._prune_rows`` with
  the shipped ``sd1.pack_semantic_state`` packer.  Every blob this module emits
  is cross-checked against ``b2e.build_edit`` tensor-by-tensor before it is
  spliced, so a drift between the two paths fails closed.
* the splice + runtime staging -- ``mp2.build_generation``, which packs the RX1M
  member, emits a deterministic ZIP, stages the candidate-bound runtime tree,
  and proves the shipped receiver reconstructs the packer's exact state.

THE BASE INSTRUMENT (binding, and the reason the ratio is fair)
--------------------------------------------------------------
The admission bar is a RATIO -- ``pose_edited / pose_base`` for the burn-2 model
against the same ratio for the calibration model.  The burn-2 base therefore has
to be the object the burn-2 edits are edits OF: the **q4-packed deployment
state**, not the float EMA.  The trainer's own parity gate validated the flat
legacy-int4 export (``pack_semantic_pose.pack_semantic``); ``sd1``'s
``legacy_int4=True`` layout is that same byte format, and its ``legacy_int4=False``
layout is the identical payload behind a 14-byte SD1M header.  Both dequantize to
the byte-identical realized state -- this module ASSERTS that -- so shipping the
SD1M-tagged carrier costs 14 raw bytes and buys a receiver the shipped runtime can
actually parse.  All four archives then ride one receiver and one quantizer, which
is what makes the four pose rows comparable.

CONTROL FIRST
-------------
Before any candidate is built, ``control_base_archive_byte_identity`` re-splices
the UNEDITED hv1 semantic section back through the same pack/zip path and refuses
unless the result is byte-identical to the pinned frontier archive.  A splice that
cannot reproduce the base is a broken instrument, and a broken instrument may not
produce a verdict.

NON-AUTHORITY
-------------
Nothing here is a score.  This module emits bytes; the pose numbers come from a
separately fired advisory decode+score, and only the exact contest evaluator on
exact archive bytes produces a score.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

b2e = importlib.import_module("experiments.ddm_b2e_edit_replay_admission")
mp2 = importlib.import_module("experiments.ddm_mp2_mixed_precision_receiver_close")
rx1 = importlib.import_module("experiments.ddm_rx1_rate_representation_attack")
sd1 = importlib.import_module("experiments.ddm_sd1_semantic_rd_curve")

SCHEMA = "ddm_b2e_admission_archives.v1"
AXIS = "[byte-only scorer-free archive splice]"
SEED = 20260816

#: The burn-2 F2-alone window this admission measures.
RUN_ROOT = Path("/Volumes/APDataStore/pact/ddm_b2e_f2_alone_run")
CHECKPOINT = RUN_ROOT / "final.pt"
CHECKPOINT_SHA256 = "464d086dad62720f9a9a32a7deb5a823d7f415a648f3345ff5b59999a8bf32db"
DEFAULT_OUTPUT = RUN_ROOT / "admission_archives"

#: The burn-2 base row -- the q4-packed deployment state, unedited.
BASE_CANDIDATE_ID = "burn2_base_q4"

#: Every candidate rides the SD1M receiver; see the module docstring.
CANDIDATE_PARSER = "sd1m"

#: The legacy flat int4 blob the trainer's parity gate validated, for the record.
LEGACY_EXPORT_BYTES = 40_252

CANDIDATE_ORDER: tuple[str, ...] = (BASE_CANDIDATE_ID, *b2e.EDIT_NAMES)


class B2EArchiveError(RuntimeError):
    """Fail-closed error for a b2e admission-archive custody or splice defect."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sweep_appledouble(root: Path) -> int:
    """Delete ExFAT AppleDouble sidecars, which break the runtime's tree scans."""

    if not root.exists():
        return 0
    before = subprocess.run(
        ["/usr/bin/find", str(root), "-name", "._*"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    if before:
        subprocess.run(
            ["/usr/bin/find", str(root), "-name", "._*", "-delete"],
            check=True,
        )
    return len(before)


# ---------------------------------------------------------------------------
# the control: the splice must reproduce the pinned frontier archive exactly
# ---------------------------------------------------------------------------


def control_base_archive_byte_identity(base_archive: Path) -> dict[str, Any]:
    """Re-splice the UNEDITED hv1 sections and refuse unless bytes are identical.

    This runs before any candidate is built.  It proves the pack/zip path this
    module drives is the one that produced the pinned frontier bytes, so a later
    byte delta is attributable to the semantic section and to nothing else.
    """

    member = mp2.read_stored_member(base_archive)
    parts = mp2.split_member(member)
    model = rx1.pack_rx1_model(
        bytes(parts["hpac"]),
        bytes(parts["semantic"]),
        bytes(parts["carrier"]),
        codec_id=rx1.RX1_CODEC_BROTLI,
        table_mode=rx1.RX1_TABLE_ON,
    )
    remembered = model + bytes(parts["tail"])
    if remembered != member:
        raise B2EArchiveError("RX1M member does not re-pack byte-identically")
    archive = rx1.deterministic_zip(remembered)
    digest = _sha256_bytes(archive)
    if len(archive) != b2e.BASE_ARCHIVE_BYTES or digest != b2e.BASE_ARCHIVE_SHA256:
        raise B2EArchiveError(
            "base archive does not round-trip byte-identically: "
            f"{len(archive)} bytes sha256={digest}"
        )
    if int(parts["semantic_size"]) != b2e.BASE_SEMANTIC_STREAM_BYTES:
        raise B2EArchiveError("hv1 semantic stream size differs from the pin")
    return {
        "control": "unedited hv1 semantic section re-spliced through the same path",
        "member_repack_byte_identical": True,
        "archive_bytes": len(archive),
        "archive_sha256": digest,
        "matches_pinned_frontier": True,
        "section_bytes": {
            "hpac": int(parts["hpac_size"]),
            "semantic": int(parts["semantic_size"]),
            "carrier": int(parts["carrier_size"]),
            "tail": len(bytes(parts["tail"])),
        },
    }


# ---------------------------------------------------------------------------
# blob construction (delegating to the harness, never reimplementing it)
# ---------------------------------------------------------------------------


def _assert_states_equal(
    expected: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> None:
    if tuple(expected) != tuple(actual):
        raise B2EArchiveError(f"{label}: tensor set or ordering differs")
    for name in expected:
        if not torch.equal(expected[name], actual[name]):
            delta = float((expected[name] - actual[name]).abs().max())
            raise B2EArchiveError(f"{label}: tensor {name} differs; max_abs={delta}")


def base_semantic_blob(
    state: Mapping[str, torch.Tensor],
) -> tuple[bytes, OrderedDict[str, torch.Tensor], dict[str, Any]]:
    """Pack the burn-2 deployment state at q4, and bind it to the trainer export.

    Emits the SD1M-tagged carrier (what the shipped receiver parses) and proves it
    dequantizes to exactly the same realized state as the flat legacy-int4 blob the
    trainer's parity gate validated.
    """

    allocation = OrderedDict((name, 4) for name in sd1.quantized_names(state))
    legacy_blob, legacy_state = sd1.pack_semantic_state(
        OrderedDict(state), allocation, legacy_int4=True
    )
    blob, realized = sd1.pack_semantic_state(
        OrderedDict(state), allocation, legacy_int4=False
    )
    _assert_states_equal(legacy_state, realized, label="legacy-int4 vs SD1M carrier")
    if len(legacy_blob) != LEGACY_EXPORT_BYTES:
        raise B2EArchiveError(
            f"legacy int4 export is {len(legacy_blob)} bytes, expected {LEGACY_EXPORT_BYTES}"
        )
    return blob, realized, {
        "construction": "sd1.pack_semantic_state, uniform q4 (the deployment pack)",
        "legacy_int4_export_bytes": len(legacy_blob),
        "legacy_int4_export_sha256": _sha256_bytes(legacy_blob),
        "sd1m_carrier_bytes": len(blob),
        "realized_state_identical_to_legacy_export": True,
        "packed_semantic_raw_bytes": len(blob),
    }


def edit_semantic_blob(
    name: str, state: Mapping[str, torch.Tensor]
) -> tuple[bytes, OrderedDict[str, torch.Tensor], dict[str, Any]]:
    """Return the packed blob for one replayed edit, cross-checked against b2e."""

    if name == "mixed_q3q4":
        blob, realized = b2e.build_mixed_q3q4(state)
    elif name in ("film_row_prune_keep87", "film_row_prune_keep75_minus_keep87"):
        ladder = name.replace("film_row_prune_", "")
        pruned = b2e._prune_rows(state, drop=ladder)
        allocation = OrderedDict((tensor, 4) for tensor in sd1.quantized_names(pruned))
        blob, realized = sd1.pack_semantic_state(
            OrderedDict(pruned), allocation, legacy_int4=False
        )
    else:
        raise B2EArchiveError(f"unknown edit: {name}")

    reference, record = b2e.build_edit(name, state)
    _assert_states_equal(reference, realized, label=f"{name} vs b2e.build_edit")
    if len(blob) != int(record["packed_semantic_raw_bytes"]):
        raise B2EArchiveError(
            f"{name}: blob is {len(blob)} bytes, b2e reports "
            f"{record['packed_semantic_raw_bytes']}"
        )
    return blob, realized, dict(record)


def semantic_blob_for(
    name: str, state: Mapping[str, torch.Tensor]
) -> tuple[bytes, OrderedDict[str, torch.Tensor], dict[str, Any]]:
    if name == BASE_CANDIDATE_ID:
        return base_semantic_blob(state)
    return edit_semantic_blob(name, state)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def build(
    output: Path,
    *,
    checkpoint: Path,
    base_archive: Path,
    candidates: Sequence[str],
) -> dict[str, Any]:
    if not str(output.resolve()).startswith("/Volumes/APDataStore/pact/"):
        raise B2EArchiveError("admission archives must be retained on APDataStore")
    control = control_base_archive_byte_identity(base_archive)

    observed_checkpoint_sha = mp2.sha256_file(checkpoint)
    if observed_checkpoint_sha != CHECKPOINT_SHA256:
        raise B2EArchiveError(
            f"burn-2 checkpoint sha256 differs from the pin: {observed_checkpoint_sha}"
        )

    template, template_provenance = b2e.load_base_state(base_archive)
    state, state_provenance = b2e.load_checkpoint_state(checkpoint, list(template))
    if state_provenance.get("deployment_weights") != "ema_shadow":
        raise B2EArchiveError(
            "burn-2 checkpoint does not declare the EMA shadow as its deployment weights"
        )
    for name in template:
        if tuple(template[name].shape) != tuple(state[name].shape):
            raise B2EArchiveError(f"burn-2 state shape differs from the template: {name}")

    member = mp2.read_stored_member(base_archive)
    base_parts = mp2.split_member(member)

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        blob, realized, construction = semantic_blob_for(candidate, state)
        stream = brotli.compress(blob, quality=11)
        if brotli.decompress(stream) != blob:
            raise B2EArchiveError(f"{candidate}: semantic Brotli parse-back differs")
        parsed, _, format_name = sd1.unpack_semantic_state(blob, state)
        if format_name != "sd1_mixed_v1":
            raise B2EArchiveError(f"{candidate}: blob is not the SD1M carrier")
        _assert_states_equal(realized, parsed, label=f"{candidate} independent parse-back")

        receipt = mp2.build_generation(
            output,
            candidate_id=candidate,
            parser=CANDIDATE_PARSER,
            semantic_stream=stream,
            semantic_raw=blob,
            expected_delta_bytes=None,
            base_parts=base_parts,
            template=template,
        )
        destination = output / "generations" / candidate
        # (review-fix) build_generation short-circuits on an existing receipt, so a
        # generation built from a DIFFERENT blob would be reused silently.  Bind the
        # retained payload to the blob this run actually computed.
        retained_raw = receipt["retained_payloads"]["semantic_raw"]
        retained_stream = receipt["retained_payloads"]["semantic_stream"]
        if str(retained_raw["sha256"]) != _sha256_bytes(blob):
            raise B2EArchiveError(
                f"{candidate}: retained semantic payload differs from the recomputed blob"
            )
        if str(retained_stream["sha256"]) != _sha256_bytes(stream):
            raise B2EArchiveError(
                f"{candidate}: retained semantic stream differs from the recomputed stream"
            )
        sweep_appledouble(destination)
        rows.append(
            {
                "candidate_id": candidate,
                "role": "base" if candidate == BASE_CANDIDATE_ID else "edit",
                "construction": construction,
                "semantic_raw_bytes": len(blob),
                "semantic_raw_sha256": _sha256_bytes(blob),
                "semantic_stream_bytes": len(stream),
                "semantic_stream_sha256": _sha256_bytes(stream),
                "semantic_stream_delta_bytes_vs_hv1": len(stream)
                - b2e.BASE_SEMANTIC_STREAM_BYTES,
                "archive": receipt["archive"],
                "archive_delta_bytes_vs_hv1": receipt["archive_delta_bytes_vs_hv1"],
                "projected_rate_only_delta_score": receipt["projected_rate_only_delta_score"],
                "generation_dir": str(destination.resolve()),
                "inflate_sh": str((destination / "inflate.sh").resolve()),
                "receiver_closed": receipt["receiver_closed"],
                "semantic_state_exact_to_packer": receipt["receiver_parseback"][
                    "semantic_state_exact_to_packer"
                ],
                "semantic_tensor_denominator": receipt["receiver_parseback"][
                    "semantic_tensor_denominator"
                ],
            }
        )

    by_id = {row["candidate_id"]: row for row in rows}
    base_row = by_id.get(BASE_CANDIDATE_ID)
    for row in rows:
        if base_row is None or row["candidate_id"] == BASE_CANDIDATE_ID:
            row["archive_delta_bytes_vs_burn2_base"] = 0 if base_row else None
            continue
        row["archive_delta_bytes_vs_burn2_base"] = (
            int(row["archive"]["bytes"]) - int(base_row["archive"]["bytes"])
        )

    result = {
        "schema": SCHEMA,
        "generated_utc": _utc(),
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "seed": SEED,
        "object_scope": (
            "archive 'semantic' section == 38-tensor SemanticTokenRenderer; the "
            "'hpac', 'carrier' and tail sections are the unedited hv1 bytes"
        ),
        "base_archive_control": control,
        "base_archive": mp2.file_record(base_archive),
        "checkpoint": {
            **mp2.file_record(checkpoint),
            "state_dict_key": state_provenance["state_dict_key"],
            "deployment_weights": state_provenance["deployment_weights"],
            "tensor_count": state_provenance["tensor_count"],
        },
        "template_provenance": template_provenance,
        "builder_source": mp2.file_record(Path(__file__)),
        "harness_source": mp2.file_record(REPO / "experiments/ddm_b2e_edit_replay_admission.py"),
        "receiver_source": mp2.file_record(mp2.RECEIVER_SOURCE),
        "candidate_denominator": len(rows),
        "candidates": rows,
        "all_receivers_closed": all(row["receiver_closed"] for row in rows),
        "complete": True,
    }
    mp2.atomic_json(output / "B2E_ADMISSION_ARCHIVES.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--base-archive", type=Path, default=b2e.BASE_ARCHIVE)
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=CANDIDATE_ORDER,
        default=list(CANDIDATE_ORDER),
    )
    parser.add_argument(
        "--control-only",
        action="store_true",
        help="run the byte-identity control and exit without building candidates",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.resolve() != args.resume_from.resolve():
        raise B2EArchiveError("--resume-from must equal --output")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    if args.control_only:
        control = control_base_archive_byte_identity(args.base_archive)
        print(json.dumps(control, indent=2, sort_keys=True))
        return
    result = build(
        args.output,
        checkpoint=args.checkpoint,
        base_archive=args.base_archive,
        candidates=args.candidates,
    )
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "candidates": [
                    {
                        "candidate_id": row["candidate_id"],
                        "archive_bytes": row["archive"]["bytes"],
                        "archive_sha256": row["archive"]["sha256"],
                        "archive_delta_bytes_vs_hv1": row["archive_delta_bytes_vs_hv1"],
                        "archive_delta_bytes_vs_burn2_base": row[
                            "archive_delta_bytes_vs_burn2_base"
                        ],
                        "receiver_closed": row["receiver_closed"],
                    }
                    for row in result["candidates"]
                ],
                "all_receivers_closed": result["all_receivers_closed"],
                "complete": result["complete"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
