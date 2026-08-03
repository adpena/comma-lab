# SPDX-License-Identifier: MIT
"""ddm_cx1 — full-clip frame parity between a legacy 6-member archive and its container.

WHY THIS EXISTS AS A TOOL
-------------------------
The whole advantage of the container line is that it is EXACTLY lossless, so
``d_seg`` and ``d_pose`` are invariant *by construction* and a rate-only ΔS needs
no scorer time.  That claim is only worth anything if it is CHECKED on the real
receiver over the whole clip: a spot check cannot distinguish "bit-identical" from
"identical on the pairs I happened to sample".  ``ddm_ix2`` ran that check from a
scratch script; this lands it, because the next composition on this line will need
it again and re-deriving a verification harness is the rediscovery sin.

WHAT IT PROVES
--------------
1. Every decoder STATE array is equal (``p_best``/``st_idx``/``sel``/``ab``/
   ``beta_idx``/``st_vals``/``beta_mags``/``dim0_offset``/``n_pairs``).
2. The TR1 packet bytes are byte-identical, so the render inputs are the same
   object and not merely equivalent-looking.
3. Every ``frame_0`` and every ``frame_1`` is byte-identical, for all pairs.

A mismatch here is a DEFECT in the container, never a result: the composition is
supposed to change bytes and nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final

import numpy as np

_REPO_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_TREE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/pfs1"
)

_STATE_KEYS: Final = (
    "n_pairs",
    "dim0_offset",
    "beta_mags",
    "st_vals",
    "p_best",
    "st_idx",
    "sel",
    "ab",
    "beta_idx",
)


class FrameParityError(AssertionError):
    """Raised when the container decodes to anything other than the legacy frames."""


def _import_receiver(runtime_tree: Path) -> Any:
    for entry in (str(runtime_tree), str(_REPO_ROOT / "experiments")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import inflate_runner_v4d  # noqa: PLC0415

    return inflate_runner_v4d


def _extract(zip_path: Path, into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = Path(info.filename)
            if name.is_absolute() or ".." in name.parts:
                raise FrameParityError(f"unsafe member name {info.filename!r}")
            archive.extract(info, into)
    return into


def _packet_equal(left: Any, right: Any) -> dict[str, bool]:
    """Compare the PARSED TR1 packet, field by field.

    Deliberately not a ``hasattr`` probe returning ``None`` when the attribute is
    missing: a check that can only ever answer "yes" or "unknown" is the vacuity
    trap — an empty scope must never emit the same symbol as a clean one.  Every
    field the renderer actually consumes is named here, so an absent one is an
    ``AttributeError`` and the tool fails loudly rather than reporting ``null``.
    """

    out = {
        "metadata": left.metadata == right.metadata,
        "selector": left.selector == right.selector,
        "section_payloads": tuple(left.section_payloads)
        == tuple(right.section_payloads),
        "token_codes": bool(np.array_equal(left.token_codes, right.token_codes)),
        "pose_stub_consumed": left.pose_stub_consumed == right.pose_stub_consumed,
    }
    for name in ("masks", "gains", "biases"):
        a, b = getattr(left, name), getattr(right, name)
        out[name] = len(a) == len(b) and all(
            np.array_equal(x, y) for x, y in zip(a, b, strict=True)
        )
    return out


def _state_equal(left: Any, right: Any) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for key in _STATE_KEYS:
        a, b = getattr(left, key), getattr(right, key)
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            out[key] = bool(np.array_equal(np.asarray(a), np.asarray(b)))
        else:
            out[key] = a == b
    return out


def compare(
    legacy_zip: Path,
    container_zip: Path,
    *,
    runtime_tree: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """Decode both archives and compare state, packet bytes and every frame."""

    receiver = _import_receiver(runtime_tree)
    with TemporaryDirectory(prefix="cx1_parity_") as scratch:
        root = Path(scratch)
        legacy_dir = _extract(legacy_zip, root / "legacy")
        container_dir = _extract(container_zip, root / "container")
        if not (container_dir / receiver.IX2_MEMBER).exists():
            raise FrameParityError(
                f"container archive has no {receiver.IX2_MEMBER}: the receiver "
                "would silently take the legacy path and the check would be vacuous"
            )
        legacy = receiver.Decoder(legacy_dir)
        container = receiver.Decoder(container_dir)

        state = _state_equal(legacy, container)
        if not all(state.values()):
            raise FrameParityError(f"decoder state differs: {state}")
        packet = _packet_equal(legacy.packet, container.packet)
        if not all(packet.values()):
            raise FrameParityError(f"parsed TR1 packet differs: {packet}")

        total = int(legacy.n_pairs) if limit is None else min(limit, legacy.n_pairs)
        mismatched: list[dict[str, Any]] = []
        for index in range(total):
            f1a, f1b = legacy.f1(index), container.f1(index)
            f0a = legacy.f0(index, f1a)
            f0b = container.f0(index, f1b)
            same1 = np.array_equal(f1a, f1b)
            same0 = np.array_equal(f0a, f0b)
            if not (same0 and same1):
                mismatched.append(
                    {"pair": index, "frame_0_equal": same0, "frame_1_equal": same1}
                )
    return {
        "schema": "ddm_cx1_frame_parity.v1",
        "legacy_archive": str(legacy_zip),
        "container_archive": str(container_zip),
        "pairs_compared": total,
        "state_equal": state,
        "parsed_packet_equal": packet,
        "mismatched_pairs": mismatched,
        "all_frames_bit_identical": not mismatched,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-archive", required=True, type=Path)
    parser.add_argument("--container-archive", required=True, type=Path)
    parser.add_argument("--runtime-tree", type=Path, default=DEFAULT_RUNTIME_TREE)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="compare only the first N pairs (a SMOKE; the verdict needs the clip)",
    )
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args(argv)

    report = compare(
        args.legacy_archive,
        args.container_archive,
        runtime_tree=args.runtime_tree,
        limit=args.limit,
    )
    report["is_full_clip"] = args.limit is None
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(text)
    print(text)
    return 0 if report["all_frames_bit_identical"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
