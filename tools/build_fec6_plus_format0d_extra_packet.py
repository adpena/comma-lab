#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# DETERMINISTIC_COMPILER_OK:fec6_stacking_wave_research_scaffold_packet_extender_per_lane_fec6_stacking_wave_5_grammar_extensions_20260517_orphan_recipe_research_only_dispatch_disabled_per_catalog_240_recipe_research_recipe_pattern_per_comprehensive_bug_audit_cascade_20260526
"""Build a fec6 + format0d-EXTRA additive latent correction packet.

This tool extends an existing fec6 archive with a format0d-EXTRA
trailing slot (Ext 4 of the fec6 stacking wave; lane
``lane_fec6_stacking_wave_5_grammar_extensions_20260517``).

The fec6 archive is a single-ZIP-member packet (member name ``x``)
containing the OUTER_MAGIC FP11 wrapper. This tool reads the existing
archive, appends the format0d-EXTRA slot to the inner-member bytes
via ``tac.fec6_format0d_extra.wrap_fec6_archive_with_extra``, and
emits a new byte-deterministic scaffold archive. The generated runtime is
intentionally scaffold-only until the fec6 base inflate path is wired; the
manifest therefore fails closed for provider and exact-eval dispatch.

Per Catalog #158 (byte-deterministic compiler): same inputs → same
output bytes.
Per Catalog #295 (self-contained inflate.py): emits an updated
inflate.py scaffold that can host the format0d-EXTRA slot decode in addition
to the existing fec6 FES1 selector once Phase 2 base-inflate wiring lands.
Per Catalog #270 (dispatch_kind: tool scope fix): this is a tool
dispatch (CPU one-shot encoder), NOT a substrate training dispatch.

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag":
predicted ΔS band ``[-0.0005, -0.0030] [predicted, council-consensus]``
on contest-CPU axis. No contest-CPU / contest-CUDA claims emitted.

Per CLAUDE.md "Apples-to-apples evidence discipline": build manifest
records source-archive sha256, output-archive sha256, format0d-EXTRA
slot sha256, per-pair (dim, delta_q) correction count, and explicit
scaffold-only dispatch blockers.

Design memo: ``.omx/research/fec6_plus_format0d_extra_design_20260517.md``
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.fec6_format0d_extra import (  # noqa: E402
    DEFAULT_DELTA_SCALE,
    EXTRA_MAGIC,
    NO_OP_DIM,
    encode_format0d_extra_payload,
    wrap_fec6_archive_with_extra,
)
from tac.repo_io import sha256_bytes, sha256_file  # noqa: E402

LANE_ID = "lane_fec6_stacking_wave_5_grammar_extensions_20260517"
MANIFEST_SCHEMA = "fec6_plus_format0d_extra_packet_manifest_v1"


def _read_single_zip_member(archive_path: Path) -> tuple[str, bytes]:
    """Read the single-member fec6 ZIP archive and return (member_name, bytes)."""
    with zipfile.ZipFile(archive_path, "r") as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise SystemExit(
                f"fec6 archive {archive_path} expected exactly 1 ZIP member; got {len(names)}: {names}"
            )
        member = names[0]
        with zf.open(member) as fh:
            payload = fh.read()
    return member, payload


def _write_single_zip_member_deterministic(
    *,
    output_path: Path,
    member_name: str,
    payload: bytes,
) -> None:
    """Write a byte-deterministic single-member ZIP archive per Catalog #158.

    Uses ZipInfo + writestr with fixed timestamp (1980-01-01) and ZIP_STORED
    method (no compression) to match the fec6 source-of-truth convention.
    """
    info = zipfile.ZipInfo(filename=member_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, payload)


def _read_extra_corrections_json(corrections_json: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Read the offline-discovered per-pair (dim, delta_q, scale) corrections.

    Expected JSON schema:
        {
            "n_pairs": int,
            "dim_arr": list[int],     # 0..28 or 255 NO_OP sentinel
            "delta_q_arr": list[int], # signed int8
            "scale": float (default 0.01),
        }
    """
    if not corrections_json.is_file():
        raise SystemExit(f"--extra-corrections-json not found: {corrections_json}")
    raw = json.loads(corrections_json.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(
            f"{corrections_json}: expected JSON object; got {type(raw).__name__}"
        )
    for key in ("dim_arr", "delta_q_arr"):
        if key not in raw:
            raise SystemExit(f"{corrections_json}: missing required key {key!r}")
    dim_arr = np.array(raw["dim_arr"], dtype=np.uint8)
    delta_q_arr = np.array(raw["delta_q_arr"], dtype=np.int8)
    scale = float(raw.get("scale", DEFAULT_DELTA_SCALE))
    if len(dim_arr) != len(delta_q_arr):
        raise SystemExit(
            f"{corrections_json}: dim_arr length {len(dim_arr)} != delta_q_arr length {len(delta_q_arr)}"
        )
    n_pairs = raw.get("n_pairs", len(dim_arr))
    if n_pairs != len(dim_arr):
        raise SystemExit(
            f"{corrections_json}: n_pairs={n_pairs} != len(dim_arr)={len(dim_arr)}"
        )
    return dim_arr, delta_q_arr, scale


# Inflate.py template that handles the format0d-EXTRA slot.
# The template wraps the existing fec6 inflate by:
#   1. Reading the fec6 inner-member bytes
#   2. Detecting the trailing EXTRA_MAGIC slot via unwrap_fec6_archive_with_extra
#   3. Stripping the extra slot, passing the base fec6 bytes to the existing
#      fec6 inflate path
#   4. After PR101's decode, applying the format0d-EXTRA per-pair latent
#      correction via the canonical apply_sidecar_corrections pattern
#
# This template MUST be edited in lockstep with submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py
# if the canonical apply_sidecar_corrections semantics change.
INFLATE_PY_TEMPLATE = r'''#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Inflate fec6 + format0d-EXTRA additive latent correction archive.

Per CLAUDE.md Catalog #205: routes through canonical select_inflate_device.
Per Catalog #295: self-contained (vendors tac.fec6_format0d_extra alongside).
"""
from __future__ import annotations

import os
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "src"
sys.path.insert(0, str(SRC_DIR))

# Vendored canonical helpers per Catalog #295.
from tac.fec6_format0d_extra import (
    EXTRA_MAGIC,
    NO_OP_DIM,
    decode_format0d_extra_payload,
    unwrap_fec6_archive_with_extra,
)


def select_inflate_device() -> torch.device:
    """Canonical inflate device selection per CLAUDE.md Catalog #205."""
    policy = os.environ.get("PACT_INFLATE_DEVICE", "auto").strip().lower()
    if policy in {"mps", "metal"}:
        raise RuntimeError(
            "PACT_INFLATE_DEVICE=mps is forbidden for auth-eval inflate; use cpu or cuda"
        )
    if policy not in {"auto", "cpu", "cuda"}:
        raise RuntimeError(
            f"PACT_INFLATE_DEVICE must be auto/cpu/cuda; got {policy!r}"
        )
    if policy == "cpu":
        return torch.device("cpu")
    if policy == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PACT_INFLATE_DEVICE=cuda requested but CUDA unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inflate(src_bin: str, dst_raw: str) -> int:
    """Inflate the fec6+format0d-EXTRA archive.

    The src_bin file is a ZIP archive with a single member; the member bytes
    are <fec6 wrapper> | <optional EXTRA slot>. We strip the EXTRA slot,
    decode it into (dim_arr, delta_q_arr, scale), then defer to the existing
    fec6 inflate path on the stripped base bytes. After PR101 decode produces
    rendered RGB frames, we ALSO apply the format0d-EXTRA correction to the
    PR101 latents (before the FES1 selector applies its frame-0 RGB
    correction).

    Returns 0 on success; raises on error.
    """
    # NOTE: this template is INCOMPLETE — operator must wire the fec6 base
    # inflate path here. Phase 1 only proves the format0d-EXTRA slot is
    # parseable and the byte-determinism contract holds. Phase 2 wiring
    # of the canonical fec6 base inflate is queued for the operator's
    # next dispatch authorization.
    raise NotImplementedError(
        "fec6+format0d-EXTRA inflate.py is a Phase 1 scaffold. "
        "The format0d-EXTRA slot decode + byte-determinism are validated by "
        "src/tac/tests/test_fec6_format0d_extra.py. Phase 2 base-inflate "
        "wiring + paired-axis Modal dispatch is queued for operator approval."
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write(f"usage: {sys.argv[0]} <src.bin> <dst.raw>\n")
        sys.exit(2)
    sys.exit(inflate(sys.argv[1], sys.argv[2]))
'''


INFLATE_SH_TEMPLATE = '''#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# fec6 + format0d-EXTRA inflate runtime entrypoint.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec uv run --with torch --with numpy "$HERE/inflate.py" "$@"
'''


def _vendor_canonical_helper(*, src_root: Path, out_runtime: Path) -> None:
    """Vendor tac.fec6_format0d_extra alongside the inflate.py per Catalog #295.

    The inflate.py imports from tac.fec6_format0d_extra; this helper copies
    the canonical module into the submission's src/tac/ subdir so the
    inflate.py works with PYTHONPATH=src on a clean Modal worker.
    """
    src_module = src_root / "src/tac/fec6_format0d_extra.py"
    if not src_module.is_file():
        raise SystemExit(f"canonical helper not found at {src_module}")
    dst_dir = out_runtime / "src" / "tac"
    dst_dir.mkdir(parents=True, exist_ok=True)
    init_py = dst_dir / "__init__.py"
    if not init_py.exists():
        init_py.write_text("", encoding="utf-8")
    shutil.copy2(src_module, dst_dir / "fec6_format0d_extra.py")


def build_packet(
    *,
    fec6_archive: Path,
    extra_corrections_json: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a fec6 + format0d-EXTRA archive and write manifest + runtime.

    Returns the build manifest dict.
    """
    fec6_archive = fec6_archive.resolve()
    extra_corrections_json = extra_corrections_json.resolve()
    output_dir = output_dir.resolve()

    if not fec6_archive.is_file():
        raise SystemExit(f"fec6 archive not found: {fec6_archive}")

    if output_dir.exists() and not overwrite:
        raise SystemExit(
            f"{output_dir} exists; pass --overwrite to replace or choose a fresh --output-dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: read the fec6 archive's single ZIP member.
    member_name, fec6_inner = _read_single_zip_member(fec6_archive)

    # Step 2: read the offline-discovered per-pair (dim, delta_q, scale) corrections.
    dim_arr, delta_q_arr, scale = _read_extra_corrections_json(extra_corrections_json)
    n_pairs = len(dim_arr)
    non_no_op_count = int((dim_arr != NO_OP_DIM).sum())

    # Step 3: encode the format0d-EXTRA payload.
    extra_payload = encode_format0d_extra_payload(
        dim_arr=dim_arr, delta_q_arr=delta_q_arr, scale=scale
    )

    # Step 4: append the EXTRA slot to the fec6 inner-member bytes.
    extended_inner = wrap_fec6_archive_with_extra(
        fec6_archive_bytes=fec6_inner, extra_payload=extra_payload
    )

    # Step 5: write the new ZIP archive (byte-deterministic).
    out_archive = output_dir / "archive.zip"
    _write_single_zip_member_deterministic(
        output_path=out_archive, member_name=member_name, payload=extended_inner
    )

    # Step 6: emit the inflate.py + inflate.sh scaffolds + vendor canonical helper.
    runtime_dir = output_dir
    (runtime_dir / "inflate.py").write_text(INFLATE_PY_TEMPLATE, encoding="utf-8")
    (runtime_dir / "inflate.sh").write_text(INFLATE_SH_TEMPLATE, encoding="utf-8")
    (runtime_dir / "inflate.sh").chmod(0o755)
    _vendor_canonical_helper(src_root=REPO_ROOT, out_runtime=runtime_dir)

    # Step 7: build the manifest with full provenance.
    extra_payload_sha = sha256_bytes(extra_payload)
    fec6_inner_sha = sha256_bytes(fec6_inner)
    extended_inner_sha = sha256_bytes(extended_inner)
    out_archive_sha = sha256_file(out_archive)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "lane_id": LANE_ID,
        "design_memo": ".omx/research/fec6_plus_format0d_extra_design_20260517.md",
        "source_fec6_archive": {
            "path": str(fec6_archive.relative_to(REPO_ROOT) if fec6_archive.is_relative_to(REPO_ROOT) else fec6_archive),
            "sha256": sha256_file(fec6_archive),
            "bytes": fec6_archive.stat().st_size,
            "inner_member_name": member_name,
            "inner_member_sha256": fec6_inner_sha,
            "inner_member_bytes": len(fec6_inner),
        },
        "extra_corrections_json": {
            "path": str(extra_corrections_json.relative_to(REPO_ROOT) if extra_corrections_json.is_relative_to(REPO_ROOT) else extra_corrections_json),
            "sha256": sha256_file(extra_corrections_json),
            "n_pairs": n_pairs,
            "non_no_op_count": non_no_op_count,
            "scale": scale,
        },
        "extra_payload": {
            "sha256": extra_payload_sha,
            "bytes": len(extra_payload),
            "magic": EXTRA_MAGIC.decode("ascii"),
        },
        "output_archive": {
            "path": str(out_archive.relative_to(REPO_ROOT) if out_archive.is_relative_to(REPO_ROOT) else out_archive),
            "sha256": out_archive_sha,
            "bytes": out_archive.stat().st_size,
            "extended_inner_member_sha256": extended_inner_sha,
            "extended_inner_member_bytes": len(extended_inner),
        },
        "byte_determinism_contract": "Catalog #158: same inputs produce identical archive bytes",
        "evidence_tag": "[predicted, council-consensus] fec6+format0d-EXTRA additive latent correction packet",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_provider_dispatch": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "runtime_scaffold_only": True,
        "runtime_consumption_proof": False,
        "byte_consumption_proof": False,
        "dispatch_blockers": [
            "inflate_py_phase1_scaffold_raises_NotImplementedError",
            "fec6_base_inflate_path_not_wired",
            "no_runtime_consumption_proof",
        ],
        "phase2_unblocker": (
            "wire generated inflate.py to the canonical fec6 base inflate path, "
            "apply format0d-EXTRA before FES1 frame-0 selector correction, then "
            "prove byte consumption and full-frame inflate success"
        ),
        "predicted_delta_s_band_contest_cpu": [-0.0030, -0.0005],
        "horizon_class": "plateau_adjacent",
    }

    (output_dir / "build_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fec6+format0d-EXTRA additive latent correction packet."
    )
    parser.add_argument(
        "--fec6-archive",
        type=Path,
        required=True,
        help="Path to the existing fec6 archive.zip (typically 6bae0201...)",
    )
    parser.add_argument(
        "--extra-corrections-json",
        type=Path,
        required=True,
        help=(
            "Path to JSON with per-pair (dim_arr, delta_q_arr, scale) from offline "
            "macOS-CPU coordinate search."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the new archive + manifest.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing --output-dir.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = build_packet(
        fec6_archive=args.fec6_archive,
        extra_corrections_json=args.extra_corrections_json,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    sys.stdout.write(
        f"[build-fec6-plus-format0d-extra] wrote {manifest['output_archive']['path']}\n"
        f"  output_archive_sha256={manifest['output_archive']['sha256']}\n"
        f"  output_archive_bytes={manifest['output_archive']['bytes']}\n"
        f"  extra_payload_bytes={manifest['extra_payload']['bytes']}\n"
        f"  non_no_op_count={manifest['extra_corrections_json']['non_no_op_count']}/{manifest['extra_corrections_json']['n_pairs']}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
