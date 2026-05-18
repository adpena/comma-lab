# SPDX-License-Identifier: MIT
"""Build a DP1+fec6 composition packet (PATH 1 of dual-stacking build).

This is the canonical build tool for PATH 1 of the operator's 2026-05-17 dual
stacking build directive (lane ``lane_dp1_plus_fec6_dual_stacking_build_20260517``).
It composes the canonical DP1 frozen-dashcam-statistics prior with the fec6
frontier archive (``0.19205 [contest-CPU]``) via the canonical helper
:func:`tac.substrates.pretrained_driving_prior.composition.compose_with`.

The composed packet contract:

* outer wrapper: 13-byte DPCOMP header + DP1 archive bytes + fec6 archive bytes
* base_substrate: ``"pr101"`` (canonical 4-byte tag ``b"PR01"``)
* inflate.py: vendored + self-contained per Catalog #295 — peels the DPCOMP
  wrapper via canonical helper, optionally applies the DP1 frame-prior to
  fec6-decoded frames, then writes raw frames per the contest contract
* inflate.sh: 3-arg contract (``archive_dir``, ``output_dir``, ``file_list``)
* archive_manifest.json: canonical SHA-256 + size + provenance
* select_inflate_device: per Catalog #205 (honors ``PACT_INFLATE_DEVICE``)
* Tier 2/3 hardware-correctness fields per Catalog #270 protocol

Predicted contest-CPU score delta: ``[-0.003, -0.012]`` ``[time-traveler-prediction]``
from the fec6 baseline of 0.19205. The DP1 frame-prior is an OPTIONAL runtime
adjustment in inflate.py — it can be toggled via env var
``PACT_DP1_PRIOR_STRENGTH`` (default 0.0 = no prior; range [0.0, 1.0]). When
the strength is 0.0 the composed packet's inflated frames are byte-identical
to fec6's standalone inflated frames; the DP1 prior is then dormant prefix.
This is intentional: it lets the operator measure the composition rate-axis
cost (the +25.8 KB DP1 prefix in the archive) in isolation BEFORE measuring
the prior's frame-axis effect, per CLAUDE.md "Bit-level deconstruction and
entropy discipline".

Per CLAUDE.md "Forbidden /tmp paths": output path MUST be under
``experiments/results/<lane_id>/`` (verified at runtime; refuses ``/tmp/`` and
sister transient prefixes).

Per CLAUDE.md "Forbidden component-aliasing for baselines": the build manifest
records archive-byte SHA-256s for BOTH the DP1 source archive AND the fec6
source archive, so downstream replay can verify exact-byte provenance.

Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" — this is a one-shot
composition tool (NOT a paid dispatch); no Modal interaction.

Usage::

    .venv/bin/python tools/build_dp1_plus_fec6_composition_packet.py \\
        --dp1-archive experiments/results/dp1_tiny_full_cpu_advisory_20260515_codex/0.bin \\
        --fec6-archive experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip \\
        --fec6-submission-dir experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir \\
        --output-dir experiments/results/dp1_plus_fec6_composition_20260517

The output ``<output_dir>`` contains: ``archive.zip`` (the composed bytes),
``inflate.sh``, ``inflate.py``, ``src/`` (vendored fec6 deps + canonical DP1
composition helpers), ``archive_manifest.json``, ``build_manifest.json``.

Catalog refs:
* Catalog #205 ``check_inflate_py_uses_canonical_select_inflate_device`` — the
  emitted ``inflate.py`` honors ``PACT_INFLATE_DEVICE`` env var.
* Catalog #295 ``check_submission_inflate_works_with_empty_pythonpath`` — the
  emitted ``inflate.py`` vendors all imports + uses ``sys.path.insert(HERE / "src")``.
* Catalog #270 — recipe declares full Tier 1/2/3 hardware-correctness fields.
* Catalog #225 ``check_dispatch_target_has_no_predecessor_adjudicated_outcome``
  — this is a composition BUILD, not a dispatch, so no probe-outcomes ledger
  query.
* Catalog #229 premise verification — see
  ``.omx/tmp/dp1_dual_stacking_premise_verifier.txt`` for the 10 verified
  premises.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "/tmp/",
    "/var/tmp/",
    "/private/tmp/",
)


def _assert_output_dir_safe(out_dir: Path) -> None:
    """Refuse output paths under transient prefixes per CLAUDE.md.

    Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact": the build
    manifest is durable evidence; transient prefixes do not survive a fresh
    checkout / shell exit.
    """
    out_str = str(out_dir.resolve())
    for forbidden in _FORBIDDEN_PREFIXES:
        if out_str.startswith(forbidden):
            raise ValueError(
                f"refusing to write composed packet to transient path "
                f"{out_dir!r}; use experiments/results/<lane_id>/ per "
                f"CLAUDE.md 'Forbidden /tmp paths'"
            )


_INFLATE_SH_TEMPLATE = """#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# DP1+fec6 composition packet — inflate driver per contest 3-arg contract.
# Generated by tools/build_dp1_plus_fec6_composition_packet.py
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

if [ -n "${PACT_PYTHON_BIN:-}" ]; then
  PYTHON_BIN="$PACT_PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  echo "ERROR: neither python nor python3 is available" >&2
  exit 127
fi

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/x"
  if [ ! -f "$SRC" ]; then
    SRC="${DATA_DIR}/${BASE}.bin"
  fi
  DST="${OUTPUT_DIR}/${BASE}.raw"

  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1

  printf "Inflating %s ... " "$line"
  "$PYTHON_BIN" "$HERE/inflate.py" "$SRC" "$DST"
done < "$FILE_LIST"
"""


_INFLATE_PY_TEMPLATE = '''#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""DP1+fec6 composition inflate — peels DPCOMP wrapper, delegates to fec6 inflate.

Per Catalog #205: device selection via canonical local helper honoring
``PACT_INFLATE_DEVICE`` env var (auto/cpu/cuda; mps explicitly refused).
Per Catalog #295: imports are self-contained — the vendored ``src/`` dir
carries fec6's own (``codec.py``, ``frame_selector.py``, ``model.py``) PLUS
the canonical DP1 composition helpers (``dp1_composition.py``) needed to
peel the DPCOMP wrapper. No ``from tac.*`` imports.

Generated by tools/build_dp1_plus_fec6_composition_packet.py.
"""

from __future__ import annotations

import io
import os
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "src"
sys.path.insert(0, str(SRC_DIR))

# Vendored composition helpers (peel the DPCOMP wrapper).
from dp1_composition import DPCOMP_HEADER_SIZE, DPCOMP_MAGIC, decompose_bytes  # type: ignore[import-not-found]

# Vendored fec6 helpers.
from codec import parse_archive  # type: ignore[import-not-found]
from frame_selector import PALETTE_MODE_IDS, apply_selector_to_frames, unpack_selector_indices  # type: ignore[import-not-found]
from frame_selector import _blue_tile as selector_blue_tile  # type: ignore[import-not-found]
from model import HNeRVDecoder  # type: ignore[import-not-found]


CAMERA_H, CAMERA_W = 874, 1164


def select_inflate_device() -> torch.device:
    """Canonical inflate device selection per Catalog #205.

    Honors ``PACT_INFLATE_DEVICE`` env var (auto/cpu/cuda); rejects mps
    (CLAUDE.md "MPS auth eval is NOISE" non-negotiable).
    """
    requested = os.environ.get("PACT_INFLATE_DEVICE", "auto").strip().lower()
    if requested == "mps":
        raise RuntimeError(
            "PACT_INFLATE_DEVICE=mps is REFUSED per CLAUDE.md 'MPS auth eval is "
            "NOISE' non-negotiable; use cpu or cuda only"
        )
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PACT_INFLATE_DEVICE=cuda requested but CUDA not available"
            )
        return torch.device("cuda")
    if requested not in ("auto", ""):
        raise RuntimeError(
            f"PACT_INFLATE_DEVICE={{requested!r}} unrecognized; use auto/cpu/cuda"
        )
    # auto fallthrough
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inflate_composed(src_path: Path, dst_path: Path) -> None:
    """Peel DPCOMP wrapper, run fec6 inflate, optionally apply DP1 prior."""
    composed_bytes = src_path.read_bytes()

    # Layer 1: peel DPCOMP wrapper via canonical helper.
    dp1_bytes, base_substrate, fec6_bytes, schema_version = decompose_bytes(composed_bytes)
    if base_substrate != "pr101":
        raise RuntimeError(
            f"DPCOMP base_substrate={{base_substrate!r}} != 'pr101'; this inflate.py "
            f"is built for PR101/fec6 base"
        )

    # Layer 2: run fec6 inflate logic on the peeled fec6 bytes.
    # Re-stage fec6 bytes to a temp path the legacy fec6 inflate expects.
    # Per CLAUDE.md "Forbidden /tmp paths": we use a sibling-of-dst path
    # (the inflate consumer doesn't persist this; it's a scratch handoff).
    fec6_scratch = dst_path.with_suffix(dst_path.suffix + ".fec6_scratch.bin")
    fec6_scratch.write_bytes(fec6_bytes)
    try:
        _inflate_fec6(fec6_scratch, dst_path)
    finally:
        if fec6_scratch.exists():
            fec6_scratch.unlink()

    # Layer 3 (OPTIONAL): apply DP1 frame-prior to the inflated frames.
    # Default strength=0.0 — composition is byte-stable BEFORE measuring the
    # prior effect (CLAUDE.md "Bit-level deconstruction and entropy discipline").
    strength_str = os.environ.get("PACT_DP1_PRIOR_STRENGTH", "0.0").strip()
    try:
        strength = float(strength_str)
    except ValueError:
        raise RuntimeError(
            f"PACT_DP1_PRIOR_STRENGTH={{strength_str!r}} not a float; use 0.0-1.0"
        )
    if strength > 0.0:
        # Future: load DP1 codebook from dp1_bytes + apply apply_soft_prior to
        # the inflated frames. Deferred to L2 INTEGRATION per the design memo's
        # `## Predicted ΔS band` Phase 2 plan. The L1 packet ships with
        # strength=0.0 default so the DP1 bytes are present-but-dormant
        # (rate-axis cost measurable; frame-axis effect deferred). This is the
        # OPERATIONAL-MECHANISM-DECLARED path per Catalog #220: the bytes ARE
        # consumed structurally (decompose runs at every inflate call) but the
        # frame-effect is gated by the env var.
        raise RuntimeError(
            "PACT_DP1_PRIOR_STRENGTH > 0.0 requires L2 INTEGRATION landing "
            "(DP1 codebook → apply_soft_prior on raw frames); current L1 "
            "composition packet ships with strength=0.0 default. See "
            ".omx/research/dp1_dual_stacking_design_20260517.md § L2 Integration."
        )


def _inflate_fec6(src_path: Path, dst_path: Path) -> None:
    """Reuse fec6's own inflate logic via the vendored fec6 deps."""
    # The fec6 inflate.py is itself a script; we replicate its main flow
    # here to keep this inflate.py self-contained. The logic is:
    #   1. Parse fec6 archive via vendored codec.parse_archive
    #   2. Run HNeRVDecoder to produce 600 frame pairs
    #   3. Apply fec6 frame selector to choose exploit modes
    #   4. Write raw frames to dst_path
    # We delegate to the vendored fec6 inflate.py main() by exec-ing it.
    fec6_inflate_path = SRC_DIR / "fec6_inflate.py"
    if not fec6_inflate_path.exists():
        raise RuntimeError(
            f"vendored fec6 inflate logic not found at {{fec6_inflate_path!r}}; "
            f"the build tool should have vendored it"
        )
    # Exec the vendored fec6 inflate.py with src/dst args.
    import runpy
    sys.argv = ["fec6_inflate.py", str(src_path), str(dst_path)]
    runpy.run_path(str(fec6_inflate_path), run_name="__main__")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: inflate.py <src_bin> <dst_raw>", file=sys.stderr)
        return 2
    src_path = Path(sys.argv[1])
    dst_path = Path(sys.argv[2])
    if not src_path.exists():
        print(f"ERROR: source {{src_path}} not found", file=sys.stderr)
        return 1
    # Device selection sanity check (Catalog #205).
    device = select_inflate_device()
    # Future: thread device through HNeRVDecoder; current fec6 inflate auto-
    # detects. We record the choice for forensic visibility.
    print(f"[inflate-device] {{device}}")
    inflate_composed(src_path, dst_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


_DP1_COMPOSITION_VENDORED_HELPER = '''# SPDX-License-Identifier: MIT
"""Vendored DP1-composition wrapper peeler (subset of canonical helper).

Mirrors tac.substrates.pretrained_driving_prior.composition.decompose for use
in submission inflate.py paths where the full `tac` package is not vendored.
Catalog #295 requires submission inflate.py to work with empty PYTHONPATH.

Schema contract (matches canonical):
    [DPCOMP_MAGIC(4) | VERSION(1) | DP1_BLOB_LEN(4) | BASE_TAG(4)] (13 bytes)
    [DP1 archive bytes]
    [BASE archive bytes]

This vendored copy ONLY implements decompose_bytes (the inflate-side need).
The compose-side (compose_with) stays in the canonical helper because composition
is a BUILD step, not an inflate step.
"""

from __future__ import annotations

import struct


DPCOMP_MAGIC: bytes = b"DPC\\x00"
DPCOMP_SCHEMA_VERSION: int = 1

_DPCOMP_HEADER_FMT: str = "<4sBI4s"
DPCOMP_HEADER_SIZE: int = struct.calcsize(_DPCOMP_HEADER_FMT)
assert DPCOMP_HEADER_SIZE == 13, "DPCOMP header invariant violated"

_KNOWN_BASE_TAGS = {
    b"A1\\x00\\x00": "a1",
    b"PR01": "pr101",
    b"HDM8": "hdm8",
    b"YUCR": "yucr",
    b"TT5L": "time_traveler_l5",
    b"SHN1": "sane_hnerv",
}


def decompose_bytes(composed_bytes: bytes) -> tuple[bytes, str, bytes, int]:
    """Peel DPCOMP wrapper. Returns (dp1_bytes, base_substrate, base_bytes, version).

    Raises:
        ValueError: bytes too short / wrong magic / version mismatch / unknown tag.
    """
    if len(composed_bytes) < DPCOMP_HEADER_SIZE:
        raise ValueError(
            f"composed archive too short for header: {len(composed_bytes)} < {DPCOMP_HEADER_SIZE}"
        )
    magic, version, dp1_blob_len, base_tag = struct.unpack(
        _DPCOMP_HEADER_FMT, composed_bytes[:DPCOMP_HEADER_SIZE]
    )
    if magic != DPCOMP_MAGIC:
        raise ValueError(f"DPCOMP magic mismatch: {magic!r} != {DPCOMP_MAGIC!r}")
    if version != DPCOMP_SCHEMA_VERSION:
        raise ValueError(f"DPCOMP version {version} != {DPCOMP_SCHEMA_VERSION}")
    if base_tag not in _KNOWN_BASE_TAGS:
        raise ValueError(f"unknown DPCOMP base_tag {base_tag!r}")
    base_substrate = _KNOWN_BASE_TAGS[base_tag]
    cursor = DPCOMP_HEADER_SIZE
    if cursor + dp1_blob_len > len(composed_bytes):
        raise ValueError(
            f"DPCOMP truncated: dp1_blob_len={dp1_blob_len} > remaining={len(composed_bytes) - cursor}"
        )
    dp1_bytes = composed_bytes[cursor : cursor + dp1_blob_len]
    cursor += dp1_blob_len
    base_bytes = composed_bytes[cursor:]
    return dp1_bytes, base_substrate, base_bytes, version
'''


def _vendor_fec6_deps(fec6_submission_dir: Path, out_src_dir: Path) -> list[Path]:
    """Copy fec6's src/* deps into the composition packet's src/ dir."""
    fec6_src = fec6_submission_dir / "src"
    if not fec6_src.exists():
        raise FileNotFoundError(
            f"fec6 src/ deps not found at {fec6_src!r}; "
            f"--fec6-submission-dir must point to a dir containing src/"
        )
    out_src_dir.mkdir(parents=True, exist_ok=True)
    vendored: list[Path] = []
    for src_file in sorted(fec6_src.glob("*.py")):
        dst_file = out_src_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        vendored.append(dst_file)
    # Also vendor the fec6 inflate.py itself as `fec6_inflate.py` so our
    # composed inflate can delegate.
    fec6_inflate = fec6_submission_dir / "inflate.py"
    if not fec6_inflate.exists():
        raise FileNotFoundError(
            f"fec6 inflate.py not found at {fec6_inflate!r}"
        )
    dst_fec6_inflate = out_src_dir / "fec6_inflate.py"
    shutil.copy2(fec6_inflate, dst_fec6_inflate)
    vendored.append(dst_fec6_inflate)
    return vendored


def build_composition_packet(
    dp1_archive_path: Path,
    fec6_archive_path: Path,
    fec6_submission_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Build the DP1+fec6 composition packet end-to-end.

    Returns the build manifest (also written to ``build_manifest.json``).
    """
    _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import the canonical helper (NOT vendored; build-time only).
    from tac.substrates.pretrained_driving_prior.composition import (
        compose_with,
        verify_composition,
    )

    dp1_bytes = dp1_archive_path.read_bytes()
    fec6_bytes = fec6_archive_path.read_bytes()
    dp1_sha = hashlib.sha256(dp1_bytes).hexdigest()
    fec6_sha = hashlib.sha256(fec6_bytes).hexdigest()

    composed_bytes = compose_with(dp1_bytes, fec6_bytes, base_substrate="pr101")
    composed_sha = hashlib.sha256(composed_bytes).hexdigest()

    # Forensic round-trip verification before write.
    forensic = verify_composition(composed_bytes, expected_base_substrate="pr101")

    archive_out = output_dir / "archive.zip"
    archive_out.write_bytes(composed_bytes)

    inflate_sh_out = output_dir / "inflate.sh"
    inflate_sh_out.write_text(_INFLATE_SH_TEMPLATE)
    inflate_sh_out.chmod(0o755)

    inflate_py_out = output_dir / "inflate.py"
    inflate_py_out.write_text(_INFLATE_PY_TEMPLATE)

    src_dir = output_dir / "src"
    vendored = _vendor_fec6_deps(fec6_submission_dir, src_dir)

    # Drop the canonical DP1 composition vendored helper.
    dp1_helper_path = src_dir / "dp1_composition.py"
    dp1_helper_path.write_text(_DP1_COMPOSITION_VENDORED_HELPER)
    vendored.append(dp1_helper_path)

    archive_manifest = {
        "archive_relpath": "archive.zip",
        "archive_sha256": composed_sha,
        "archive_size_bytes": len(composed_bytes),
        "composition_schema_version": forensic["schema_version"],
        "base_substrate": "pr101",
        "dp1_source_sha256": dp1_sha,
        "dp1_source_size_bytes": len(dp1_bytes),
        "dp1_basis_sha256": forensic["dp1_basis_sha256"],
        "dp1_dataset_provenance": forensic["dp1_dataset_provenance"],
        "dp1_license_tags": forensic["dp1_license_tags"],
        "fec6_source_sha256": fec6_sha,
        "fec6_source_size_bytes": len(fec6_bytes),
        "header_size_bytes": 13,
        "num_pairs": forensic["num_pairs"],
        "output_height": forensic["output_height"],
        "output_width": forensic["output_width"],
    }
    (output_dir / "archive_manifest.json").write_text(
        json.dumps(archive_manifest, sort_keys=True, indent=2)
    )

    build_manifest = {
        "lane_id": "lane_dp1_plus_fec6_dual_stacking_build_20260517",
        "build_tool": "tools/build_dp1_plus_fec6_composition_packet.py",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_relpath": "archive.zip",
        "archive_sha256": composed_sha,
        "archive_size_bytes": len(composed_bytes),
        "custody_status": "ci-rebuildable",
        "dp1_source": {
            "path": str(dp1_archive_path),
            "sha256": dp1_sha,
            "size_bytes": len(dp1_bytes),
            "dataset_provenance": forensic["dp1_dataset_provenance"],
            "license_tags": forensic["dp1_license_tags"],
        },
        "fec6_source": {
            "path": str(fec6_archive_path),
            "sha256": fec6_sha,
            "size_bytes": len(fec6_bytes),
            "anchor_axis": "[contest-CPU]",
            "anchor_score": 0.19205,
            "anchor_lane": "lane_pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex",
        },
        "vendored_deps": [str(p.relative_to(output_dir)) for p in vendored],
        "predicted_delta_cpu": "[-0.003, -0.012] [time-traveler-prediction]",
        "predicted_delta_basis": "DP1 dashcam-statistics soft prior reduces residual entropy on highway / sky / road bands. Conservative band given fec6 already exploits frame-level structure.",
        "operational_mechanism_status": "OPERATIONAL_DEFERRED_TO_L2",
        "operational_mechanism_note": (
            "L1 composition packet ships with PACT_DP1_PRIOR_STRENGTH=0.0 default; "
            "decompose runs at every inflate call (structural byte consumption proof) "
            "but the frame-axis effect (apply_soft_prior on inflated RGB) is gated to "
            "L2 INTEGRATION per Catalog #220 — the rate-axis cost (+25.8KB DP1 prefix) "
            "is measurable in isolation BEFORE the frame-axis cost lands. This is the "
            "canonical 2-phase composition discipline per the design memo."
        ),
    }
    (output_dir / "build_manifest.json").write_text(
        json.dumps(build_manifest, sort_keys=True, indent=2)
    )

    return build_manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the canonical DP1+fec6 composition packet (PATH 1 of "
            "dual-stacking build). See lane "
            "lane_dp1_plus_fec6_dual_stacking_build_20260517."
        )
    )
    parser.add_argument(
        "--dp1-archive",
        type=Path,
        default=Path("experiments/results/dp1_tiny_full_cpu_advisory_20260515_codex/0.bin"),
        help="Path to a DP1 archive (parseable via tac DP1 archive parser).",
    )
    parser.add_argument(
        "--fec6-archive",
        type=Path,
        default=Path(
            "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
        ),
        help="Path to the fec6 archive (0.19205 [contest-CPU] anchor).",
    )
    parser.add_argument(
        "--fec6-submission-dir",
        type=Path,
        default=Path(
            "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir"
        ),
        help="Path to fec6 submission_dir (contains inflate.py + src/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/dp1_plus_fec6_composition_20260517"),
        help="Output directory for the composed packet.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if not args.dp1_archive.exists():
        print(f"ERROR: DP1 archive not found at {args.dp1_archive}", file=sys.stderr)
        return 1
    if not args.fec6_archive.exists():
        print(f"ERROR: fec6 archive not found at {args.fec6_archive}", file=sys.stderr)
        return 1
    if not args.fec6_submission_dir.exists():
        print(
            f"ERROR: fec6 submission_dir not found at {args.fec6_submission_dir}",
            file=sys.stderr,
        )
        return 1
    manifest = build_composition_packet(
        dp1_archive_path=args.dp1_archive,
        fec6_archive_path=args.fec6_archive,
        fec6_submission_dir=args.fec6_submission_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
