#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# DETERMINISTIC_COMPILER_OK: retired torch_vehicle bc20-era G3 packet builder (June 2026 lineage); only consumer is its own test; the live shipping path is the E4/candidate-seal chain, which routes through the canonical compiler surfaces (Catalog #158 backfill 2026-08-25)
"""Assemble a SELF-CONTAINED, RUNNABLE contest submission_dir for the torch-vehicle
bc20 small-basis vehicle — the G3 dual CPU+CUDA exact-eval input packet.

WHY a dedicated builder (search-first, named):
  * ``tools/build_torch_vehicle_d2_archive_zip.py`` builds ONLY ``archive.zip`` (the
    0.bin member) — NOT the runtime tree, so it is not a runnable submission_dir.
  * ``tools/verify_e2e_byte_close_eval.py`` assembles a FLAT advisory packet but copies
    the vendored ``inflate.sh`` whose ``python -m submissions.${SUB_NAME}.inflate`` form
    is bound to the PR95 ``<ROOT>/submissions/hnerv_muon/`` layout and to the public-PR
    intake clone (which lacks ``hnerv_muon/__init__.py`` and must NOT be edited in place
    per CLAUDE.md "Forbidden in-place edits to public PR intake clones"). That advisory
    packet runs the IN-PYTHON authority path, not the contest ``inflate.sh`` subprocess.

WHAT THIS BUILDS (the contest contract, verified against ``upstream/evaluate.sh`` +
``experiments/contest_auth_eval.py::_run_inflate``):
  * ``submission_dir/archive.zip``  — ZIP_STORED member ``0.bin`` (the byte-closed payload),
    deterministic mtime, byte-identical to the basin's ``best_archive.bin``.
  * ``submission_dir/inflate.sh``   — a CLEAN direct-call runner matching the contest
    ``inflate.sh <archive_dir> <inflated_dir> <video_names>`` signature: loops the video
    list, runs ``python inflate.py <archive_dir>/<base>.bin <inflated_dir>/<base>.raw``.
    Uses the SELF-CONTAINED ``inflate.py`` (which inserts ``src/`` on ``sys.path``) — NO
    ``-m`` package import, so the packet runs from any extracted location (the Modal
    wrapper extracts it to an arbitrary ``/tmp/.../submission_dir``).
  * ``submission_dir/inflate.py``   — the vendored self-contained inflater (COPIED, not
    in-place edited; uses ``torch`` CUDA-or-CPU agnostic + bicubic upsample to 874x1164).
  * ``submission_dir/src/{model.py,codec.py}`` — the minimal dependency closure inflate.py
    imports (``HNeRVDecoder`` + ``parse_archive``). Verified: model.py / codec.py import
    only torch / numpy / brotli; no cross-imports to data/losses/optim/score/train.

The 0.bin is built by the SAME vendored ``build_archive`` the driver byte-closes with —
NO codec is reimplemented. The decoder + latents are loaded from the checkpoint dir
(``best_ema_decoder.pt`` + ``best_ema_latents.pt`` + ``best_meta.json``) exactly as
``tools/verify_e2e_byte_close_eval.py`` does, and a parse-back fixed-point check proves
the 0.bin round-trips (weights + latents bit-exact on re-parse).

AUTHORITY: the packet is contest-runnable; the SCORE is authoritative ONLY after
``upstream/evaluate.py`` on contest-compliant Linux x86_64 CPU / CUDA (the G3 exact row).
This builder makes NO score / frontier / promotion claim. ``[contest-* advisory until
exact eval]``.

Usage:
    .venv/bin/python tools/build_torch_vehicle_g3_contest_packet.py \\
        --ckpt-dir experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best \\
        --out-dir experiments/results/g3_torch_vehicle_bc20_packet_<ts>/submission_dir
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# A CLEAN direct-call inflate.sh: matches the contest `inflate.sh <archive_dir>
# <inflated_dir> <video_names>` signature (upstream/evaluate.sh + contest_auth_eval
# _run_inflate both invoke `bash inflate.sh archive/ inflated/ video_names`). Runs the
# SELF-CONTAINED inflate.py directly (no `-m` package import) so the packet runs from
# any extracted location. ${PYTHON:-python3} honors contest_auth_eval's interpreter env.
_INFLATE_SH = """#!/usr/bin/env bash
# Inflate the torch-vehicle bc20 HNeRV archive to raw uint8 RGB frames.
# Signature (contest contract): inflate.sh <archive_dir> <inflated_dir> <video_names_file>
# A .raw file is a flat (N, 874, 1164, 3) uint8 dump, no header.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

PYBIN="${PYTHON:-${PYTHON_BIN:-python3}}"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/${BASE}.bin"
  DST="${OUTPUT_DIR}/${BASE}.raw"

  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1

  mkdir -p "$(dirname "$DST")"
  printf "Inflating %s ... " "$line"
  "$PYBIN" "$HERE/inflate.py" "$SRC" "$DST"
done < "$FILE_LIST"
"""

_README = """# torch_vehicle bc20 small-basis — G3 contest packet

Self-contained HNeRV-Muon decoder vehicle (base_channels=20, latent_dim=28, 600 pairs).

* archive.zip — ZIP_STORED member 0.bin (byte-closed decoder + per-frame-pair latents)
* inflate.sh  — contest runner: archive_dir/0.bin -> decoder forward -> bicubic 874x1164
                -> uint8 -> inflated_dir/0.raw (N=1200, H=874, W=1164, 3 channels)
* inflate.py / src/{model.py,codec.py} — the self-contained runtime (torch CUDA-or-CPU)

The score is authoritative ONLY after upstream/evaluate.py on the byte-closed archive.zip
on contest-compliant Linux x86_64 CPU / CUDA (the G3 dual exact row). NON-PROMOTABLE
advisory until then.
"""


def _build_archive_bytes(ckpt_dir: Path) -> tuple[bytes, dict]:
    """Load decoder+latents+meta from the checkpoint dir and byte-close with the SAME
    vendored build_archive the driver uses. Returns (archive_bytes, sections_meta).

    REUSES the canonical ``tools/verify_e2e_byte_close_eval.py`` loader + parity check
    (NO codec / parity reimplemented): the G2 byte-close parity contract is the FIXED-POINT
    check (parse-back state re-encoded + re-parsed is bit-exact on weights + latents), NOT a
    naive original-float vs quantized parse-back equality (the codec quantizes int8 + fp16).
    A plain (non-FiLM) checkpoint takes the pristine vendored 3-section archive route.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "verify_e2e_byte_close_eval",
        _REPO_ROOT / "tools" / "verify_e2e_byte_close_eval.py",
    )
    e2e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e2e)

    from tac.torch_vehicle.driver import import_vendored_bundle

    v = import_vendored_bundle()
    dec_sd, latents, meta = e2e._load_checkpoint(ckpt_dir)

    # Plain (non-FiLM) vehicle: the bc20 basin is a bare HNeRV decoder state dict.
    is_film = e2e._is_film_wrapper_state_dict(dec_sd) if hasattr(e2e, "_is_film_wrapper_state_dict") else False
    if is_film:
        raise SystemExit(
            "FATAL: this packet builder targets the PLAIN bc20 vehicle; the checkpoint "
            "is a FiLM-v2 wrapper. Use the verify_e2e FiLM route for a pose-section packet."
        )

    n_pairs = int(latents.shape[0])
    latent_dim = int(latents.shape[1])
    base_channels = int(meta.get("base_channels", 20))
    meta_dict = {
        "n_pairs": n_pairs,
        "latent_dim": latent_dim,
        "base_channels": base_channels,
        "eval_size": list(meta.get("eval_size", [384, 512])),
    }
    archive, _dec_back, _lat_back, status = e2e._byte_close_and_verify_parity(
        v, dec_sd, latents, meta_dict
    )

    return archive, {
        "n_pairs": n_pairs,
        "latent_dim": latent_dim,
        "base_channels": base_channels,
        "eval_size": meta_dict["eval_size"],
        "archive_bytes": len(archive),
        "parse_back_parity_ok": bool(status.get("parity_ok", False)),
        "parity_status": status,
        "ckpt_meta": meta,
    }


def build_packet(ckpt_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Byte-close the archive (vendored build_archive; parse-back parity proof).
    archive, close_meta = _build_archive_bytes(ckpt_dir)
    if not close_meta["parse_back_parity_ok"]:
        raise SystemExit("FATAL: parse-back parity FAILED — 0.bin is not a fixed point; refusing to build packet")

    # 2. 0.bin (the raw payload, for byte-fact records) + archive.zip (ZIP_STORED member 0.bin).
    bin_path = out_dir / "0.bin"
    bin_path.write_bytes(archive)
    zip_path = out_dir / "archive.zip"
    info = zipfile.ZipInfo("0.bin")
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(info, archive)

    # 3. Runtime tree: clean inflate.sh + COPIED vendored inflate.py + src/{model,codec}.
    from tac.torch_vehicle.vendored_imports import VENDORED_SRC

    vendored_root = VENDORED_SRC.parent  # .../submissions/hnerv_muon
    (out_dir / "inflate.sh").write_text(_INFLATE_SH)
    (out_dir / "inflate.sh").chmod(0o755)
    shutil.copy2(vendored_root / "inflate.py", out_dir / "inflate.py")
    src_dst = out_dir / "src"
    src_dst.mkdir(exist_ok=True)
    for name in ("model.py", "codec.py"):
        shutil.copy2(VENDORED_SRC / name, src_dst / name)
    (out_dir / "README.md").write_text(_README)

    # 4. Manifest (custody: archive + 0.bin + runtime tree shas).
    archive_sha = _sha256_file(zip_path)
    runtime_files = sorted(
        str(p.relative_to(out_dir)) for p in out_dir.rglob("*")
        if p.is_file() and p.name not in ("0.bin", "archive.zip")
    )
    runtime_tree_sha = _sha256_bytes(
        b"".join(
            (rf + ":" + _sha256_file(out_dir / rf)).encode() for rf in runtime_files
        )
    )
    manifest = {
        "schema": "torch_vehicle_g3_contest_packet_manifest.v1",
        "authority": "[contest-* advisory until upstream/evaluate.py on byte-closed archive] NON-PROMOTABLE",
        "ckpt_dir": str(ckpt_dir),
        "submission_dir": str(out_dir),
        "archive_zip": str(zip_path),
        "archive_zip_bytes": zip_path.stat().st_size,
        "archive_zip_sha256": archive_sha,
        "zero_bin_bytes": bin_path.stat().st_size,
        "zero_bin_sha256": _sha256_file(bin_path),
        "zip_overhead_bytes": zip_path.stat().st_size - bin_path.stat().st_size,
        "inflate_sh_rel": "inflate.sh",
        "runtime_files": runtime_files,
        "runtime_tree_sha256": runtime_tree_sha,
        "rate_denom_default": 37_545_489,
        "contest_rate_25x_over_default_denom": round(
            25 * zip_path.stat().st_size / 37_545_489, 8
        ),
        "built_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **close_meta,
    }
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--ckpt-dir", type=Path,
        default=_REPO_ROOT / "experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best",
        help="driver best/ dir with best_ema_decoder.pt + best_ema_latents.pt + best_meta.json",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="submission_dir to write (default experiments/results/g3_torch_vehicle_bc20_packet_<ts>/submission_dir)",
    )
    ap.add_argument("--manifest-json", type=Path, default=None)
    args = ap.parse_args(argv)

    out_dir = args.out_dir or (
        _REPO_ROOT / "experiments" / "results"
        / f"g3_torch_vehicle_bc20_packet_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        / "submission_dir"
    )
    manifest = build_packet(args.ckpt_dir.resolve(), out_dir.resolve())
    text = json.dumps(manifest, indent=2, sort_keys=True)
    print(text)
    mpath = args.manifest_json or (out_dir.parent / "g3_packet_manifest.json")
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(text + "\n", encoding="utf-8")
    print(f"[packet] wrote {out_dir}  manifest={mpath}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
