#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# DETERMINISTIC_COMPILER_OK:fec6_stacking_wave_5_haar_residual_research_scaffold_packet_extender_per_lane_fec6_stacking_wave_5_grammar_extensions_20260517_orphan_recipe_research_only_dispatch_disabled_per_catalog_240_recipe_research_recipe_pattern_per_comprehensive_bug_audit_cascade_20260526
"""Build a fec6 + Haar 1-level wavelet residual packet (Ext 5).

This tool extends an existing fec6 archive with a Haar residual scaffold
trailing slot. The residual is computed offline from a precomputed
per-frame residual stack (residual = target_frame - fec6_rendered_frame
at a downsampled resolution), Haar-transformed, per-band-quantized,
and packed into the byte-deterministic wire format.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L7:
bolt-on scope (1-level Haar). Daubechies-4 multi-level is
substrate-engineering scope and is deferred.

Per Catalog #158 byte-determinism: same inputs → same output bytes.
Per Catalog #295: scaffold inflate.py vendors tac.fec6_haar_residual and
intentionally fails closed until fec6 base-inflate wiring lands.
Per Catalog #270 (dispatch_kind: tool): tool dispatch, not substrate training.

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag":
predicted ΔS band ``[-0.0005, -0.0020] [predicted, theoretical]`` on
contest-CPU axis. The current scaffold does not make that claim actionable:
the raw Haar payload is uncompressed and the runtime raises
``NotImplementedError``.

Design memo: ``.omx/research/fec6_plus_haar_residual_design_20260517.md``
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

from tac.fec6_haar_residual import (  # noqa: E402
    WAVELET_MAGIC,
    encode_haar_residual_payload,
    wrap_fec6_archive_with_haar,
)
from tac.repo_io import sha256_bytes, sha256_file  # noqa: E402

LANE_ID = "lane_fec6_stacking_wave_5_grammar_extensions_20260517"
MANIFEST_SCHEMA = "fec6_plus_haar_residual_packet_manifest_v1"

DEFAULT_FRAME_H = 874  # camera resolution height
DEFAULT_FRAME_W = 1164  # camera resolution width


def _read_single_zip_member(archive_path: Path) -> tuple[str, bytes]:
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
    info = zipfile.ZipInfo(filename=member_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, payload)


def _load_residuals_from_npz(npz_path: Path) -> np.ndarray:
    """Load per-frame residuals from a .npz file.

    Expected structure:
        - 'residuals' : np.ndarray, shape (n_frames, residual_h, residual_w)
            float32 residual at the downsampled resolution
    """
    if not npz_path.is_file():
        raise SystemExit(f"--residuals-npz not found: {npz_path}")
    with np.load(npz_path) as npz:
        if "residuals" not in npz.files:
            raise SystemExit(
                f"{npz_path}: expected 'residuals' key in npz; got {list(npz.files)}"
            )
        residuals = npz["residuals"]
    if residuals.ndim != 3:
        raise SystemExit(
            f"residuals must be 3-D (n_frames, H, W); got shape {residuals.shape}"
        )
    return residuals.astype(np.float32, copy=False)


INFLATE_PY_TEMPLATE = r'''#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Inflate fec6 + Haar residual archive.

Per CLAUDE.md Catalog #205: routes through canonical select_inflate_device.
Per Catalog #295: self-contained (vendors tac.fec6_haar_residual).

This is a Phase 1 SCAFFOLD inflate.py. The Haar residual slot decode is
fully wired and tested via src/tac/tests/test_fec6_plus_haar_residual.py.
Phase 2 base-inflate wiring (fec6 + PR101 + frame_selector) is queued for
operator approval; dispatch will refuse until Phase 2 lands.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "src"
sys.path.insert(0, str(SRC_DIR))

from tac.fec6_haar_residual import (
    WAVELET_MAGIC,
    decode_haar_residual_payload,
    dequantize_per_band,
    haar_inverse_1level,
    unwrap_fec6_archive_with_haar,
)


def select_inflate_device() -> torch.device:
    """Canonical inflate device selection per CLAUDE.md Catalog #205."""
    policy = os.environ.get("PACT_INFLATE_DEVICE", "auto").strip().lower()
    if policy in {"mps", "metal"}:
        raise RuntimeError(
            "PACT_INFLATE_DEVICE=mps is forbidden for auth-eval inflate; use cpu or cuda"
        )
    if policy not in {"auto", "cpu", "cuda"}:
        raise RuntimeError(f"PACT_INFLATE_DEVICE must be auto/cpu/cuda; got {policy!r}")
    if policy == "cpu":
        return torch.device("cpu")
    if policy == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PACT_INFLATE_DEVICE=cuda requested but CUDA unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reconstruct_haar_residuals_per_frame(decoded: dict) -> np.ndarray:
    """Inverse-Haar each frame in the decoded payload and return per-frame residuals.

    Returns np.ndarray of shape (n_frames, 2*h_ll, 2*w_ll) float32.
    """
    n_frames = decoded["n_frames"]
    out = np.empty(
        (n_frames, 2 * decoded["h_ll"], 2 * decoded["w_ll"]), dtype=np.float32
    )
    for i in range(n_frames):
        ll = dequantize_per_band(decoded["ll_quant"][i], decoded["scale_ll"])
        lh = dequantize_per_band(decoded["lh_quant"][i], decoded["scale_lh"])
        hl = dequantize_per_band(decoded["hl_quant"][i], decoded["scale_hl"])
        hh = dequantize_per_band(decoded["hh_quant"][i], decoded["scale_hh"])
        out[i] = haar_inverse_1level(ll, lh, hl, hh)
    return out


def inflate(src_bin: str, dst_raw: str) -> int:
    """Phase 1 scaffold: validate the Haar residual slot is parseable.

    Phase 2 wiring of the fec6 base inflate path is queued for operator
    approval. Until Phase 2 lands, this inflate.py refuses to produce
    output (raises NotImplementedError) so dispatch is fail-closed.
    """
    raise NotImplementedError(
        "fec6+haar-residual inflate.py is a Phase 1 scaffold. "
        "The Haar residual slot decode + per-frame inverse-Haar are validated "
        "by src/tac/tests/test_fec6_plus_haar_residual.py. Phase 2 fec6 base-inflate "
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
# fec6 + Haar residual inflate runtime entrypoint.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec uv run --with torch --with numpy "$HERE/inflate.py" "$@"
'''


def _vendor_canonical_helper(*, src_root: Path, out_runtime: Path) -> None:
    src_module = src_root / "src/tac/fec6_haar_residual.py"
    if not src_module.is_file():
        raise SystemExit(f"canonical helper not found at {src_module}")
    dst_dir = out_runtime / "src" / "tac"
    dst_dir.mkdir(parents=True, exist_ok=True)
    init_py = dst_dir / "__init__.py"
    if not init_py.exists():
        init_py.write_text("", encoding="utf-8")
    shutil.copy2(src_module, dst_dir / "fec6_haar_residual.py")


def build_packet(
    *,
    fec6_archive: Path,
    residuals_npz: Path,
    output_dir: Path,
    frame_h: int = DEFAULT_FRAME_H,
    frame_w: int = DEFAULT_FRAME_W,
    overwrite: bool = False,
) -> dict[str, Any]:
    fec6_archive = fec6_archive.resolve()
    residuals_npz = residuals_npz.resolve()
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

    # Step 2: load the offline-computed per-frame residuals.
    residuals = _load_residuals_from_npz(residuals_npz)
    n_frames, residual_h, residual_w = residuals.shape

    # Step 3: encode the Haar residual payload.
    haar_payload = encode_haar_residual_payload(
        residuals=residuals, frame_h=frame_h, frame_w=frame_w
    )

    # Step 4: append the Haar slot to the fec6 inner-member bytes.
    extended_inner = wrap_fec6_archive_with_haar(
        fec6_archive_bytes=fec6_inner, haar_payload=haar_payload
    )

    # Step 5: write the new ZIP archive byte-deterministically.
    out_archive = output_dir / "archive.zip"
    _write_single_zip_member_deterministic(
        output_path=out_archive, member_name=member_name, payload=extended_inner
    )

    # Step 6: scaffold inflate.py + inflate.sh + vendor canonical helper.
    (output_dir / "inflate.py").write_text(INFLATE_PY_TEMPLATE, encoding="utf-8")
    (output_dir / "inflate.sh").write_text(INFLATE_SH_TEMPLATE, encoding="utf-8")
    (output_dir / "inflate.sh").chmod(0o755)
    _vendor_canonical_helper(src_root=REPO_ROOT, out_runtime=output_dir)

    # Step 7: per-band stats for observability (Catalog #305).
    # Per-band L2 from forward Haar (sanity check for band-energy distribution).
    from tac.fec6_haar_residual import haar_forward_1level
    ll_e = lh_e = hl_e = hh_e = 0.0
    for i in range(n_frames):
        ll, lh, hl, hh = haar_forward_1level(residuals[i])
        ll_e += float((ll.astype(np.float64) ** 2).sum())
        lh_e += float((lh.astype(np.float64) ** 2).sum())
        hl_e += float((hl.astype(np.float64) ** 2).sum())
        hh_e += float((hh.astype(np.float64) ** 2).sum())

    haar_payload_sha = sha256_bytes(haar_payload)
    fec6_inner_sha = sha256_bytes(fec6_inner)
    extended_inner_sha = sha256_bytes(extended_inner)
    out_archive_sha = sha256_file(out_archive)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "lane_id": LANE_ID,
        "design_memo": ".omx/research/fec6_plus_haar_residual_design_20260517.md",
        "source_fec6_archive": {
            "path": str(fec6_archive.relative_to(REPO_ROOT) if fec6_archive.is_relative_to(REPO_ROOT) else fec6_archive),
            "sha256": sha256_file(fec6_archive),
            "bytes": fec6_archive.stat().st_size,
            "inner_member_name": member_name,
            "inner_member_sha256": fec6_inner_sha,
            "inner_member_bytes": len(fec6_inner),
        },
        "residuals_npz": {
            "path": str(residuals_npz.relative_to(REPO_ROOT) if residuals_npz.is_relative_to(REPO_ROOT) else residuals_npz),
            "sha256": sha256_file(residuals_npz),
            "n_frames": n_frames,
            "residual_h": residual_h,
            "residual_w": residual_w,
        },
        "haar_payload": {
            "sha256": haar_payload_sha,
            "bytes": len(haar_payload),
            "magic": WAVELET_MAGIC.decode("ascii"),
        },
        "output_archive": {
            "path": str(out_archive.relative_to(REPO_ROOT) if out_archive.is_relative_to(REPO_ROOT) else out_archive),
            "sha256": out_archive_sha,
            "bytes": out_archive.stat().st_size,
            "extended_inner_member_sha256": extended_inner_sha,
            "extended_inner_member_bytes": len(extended_inner),
        },
        "per_band_energy": {
            "ll": ll_e,
            "lh": lh_e,
            "hl": hl_e,
            "hh": hh_e,
            "total": ll_e + lh_e + hl_e + hh_e,
            "comment": "orthonormal Haar preserves L2; per-band stats for observability per Catalog #305",
        },
        "byte_determinism_contract": "Catalog #158: same inputs produce identical archive bytes",
        "evidence_tag": "[predicted, theoretical] fec6+Haar 1-level wavelet residual packet",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_provider_dispatch": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "runtime_scaffold_only": True,
        "runtime_consumption_proof": False,
        "byte_consumption_proof": False,
        "payload_compression": "none_raw_int8_phase1_scaffold",
        "dispatch_blockers": [
            "inflate_py_phase1_scaffold_raises_NotImplementedError",
            "fec6_base_inflate_path_not_wired",
            "no_runtime_consumption_proof",
            "haar_payload_not_entropy_compressed",
        ],
        "phase2_unblocker": (
            "wire generated inflate.py to canonical fec6 base inflate, apply "
            "decoded Haar residual to rendered frames, prove byte consumption "
            "and full-frame inflate success, then add a measured entropy coder "
            "if raw payload bytes exceed the marginal score budget"
        ),
        "predicted_delta_s_band_contest_cpu": [-0.0020, -0.0005],
        "horizon_class": "plateau_adjacent",
        "frame_dimensions": {"frame_h": frame_h, "frame_w": frame_w},
    }

    (output_dir / "build_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fec6+Haar residual packet (Ext 5 of fec6 stacking wave)."
    )
    parser.add_argument(
        "--fec6-archive",
        type=Path,
        required=True,
        help="Path to the existing fec6 archive.zip",
    )
    parser.add_argument(
        "--residuals-npz",
        type=Path,
        required=True,
        help=(
            "Path to npz file with 'residuals' key: shape (n_frames, residual_h, residual_w) "
            "float32 residual computed offline from (target_frame - fec6_rendered_frame) "
            "at downsampled resolution."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the new archive + manifest.",
    )
    parser.add_argument(
        "--frame-h",
        type=int,
        default=DEFAULT_FRAME_H,
        help=f"Full camera frame height (default {DEFAULT_FRAME_H}).",
    )
    parser.add_argument(
        "--frame-w",
        type=int,
        default=DEFAULT_FRAME_W,
        help=f"Full camera frame width (default {DEFAULT_FRAME_W}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing --output-dir.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = build_packet(
        fec6_archive=args.fec6_archive,
        residuals_npz=args.residuals_npz,
        output_dir=args.output_dir,
        frame_h=args.frame_h,
        frame_w=args.frame_w,
        overwrite=args.overwrite,
    )
    sys.stdout.write(
        f"[build-fec6-plus-haar-residual] wrote {manifest['output_archive']['path']}\n"
        f"  output_archive_sha256={manifest['output_archive']['sha256']}\n"
        f"  output_archive_bytes={manifest['output_archive']['bytes']}\n"
        f"  haar_payload_bytes={manifest['haar_payload']['bytes']}\n"
        f"  n_frames={manifest['residuals_npz']['n_frames']} "
        f"residual={manifest['residuals_npz']['residual_h']}x{manifest['residuals_npz']['residual_w']}\n"
        f"  per_band_energy_total={manifest['per_band_energy']['total']:.3f} "
        f"(LL={manifest['per_band_energy']['ll']:.3f}, "
        f"LH={manifest['per_band_energy']['lh']:.3f}, "
        f"HL={manifest['per_band_energy']['hl']:.3f}, "
        f"HH={manifest['per_band_energy']['hh']:.3f})\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
