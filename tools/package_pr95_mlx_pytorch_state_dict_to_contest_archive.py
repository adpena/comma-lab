#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Package a PR95/HNeRV PyTorch state_dict into a byte-closed contest archive.

This is canonical loop closure cascade piece #2 per the Slot 4 PR95-MLX loop
closure cascade plan (`.omx/research/pr95_mlx_loop_closure_cascade_plan_*`).

Slot 1 (`tools/export_pr95_mlx_to_pytorch_state_dict.py`) emits a PyTorch
state_dict `.pt` whose forward-parity against the public PR95 decoder is proven
locally on macOS via MLX. This tool consumes that `.pt` AND the source PR95
public archive zip (canonical source of the latents tensor + meta), and writes
a byte-closed contest archive matching the canonical PR95 archive grammar
(single member `0.bin` = meta_blob | decoder_blob | latents_blob per the PR95
`build_archive` source).

Heavy lifting delegated to the canonical helper
`tac.local_acceleration.pr95_hnerv_mlx.write_pr95_public_archive_zip` which:
  - quantizes state_dict tensors per-tensor symmetric INT8 (canonical PR95
    `quantize_state_dict`),
  - encodes the latents per-dim asymmetric UINT8 + 1st-order temporal delta +
    zigzag + lo/hi split (canonical PR95 `encode_latents`),
  - brotli-compresses each blob (quality=11),
  - emits a single-member deterministic ZIP (ZIP_STORED, fixed timestamp).

The output is intentionally [macOS-CPU advisory only] until paired contest
CPU + CUDA auth eval lands per CLAUDE.md "Submission auth eval - BOTH CPU AND
CUDA" non-negotiable + Catalog #192/#317/#341 canonical-routing-markers
discipline.

CANONICAL DESIGN DECISIONS (from canonical PR95 source + canonical helper):
  - renderer.bin: INT8 quantized state_dict + brotli, NOT FP4+brotli (this is
    canonical PR95; Quantizr 0.33 uses FP4+brotli but PR95 hnerv_muon does
    NOT).
  - latents.bin: LEARNED end-to-end via training; round-tripped from the
    source archive zip (the .pt does NOT carry latents because it is decoder-
    state-only per Slot 1 contract).
  - masks.mkv: NOT EMITTED. The PR95/HNeRV archive grammar does not include
    masks.mkv because the renderer outputs full RGB frames and the contest
    scorer derives masks from the rendered frames. (The Quantizr 0.33
    paradigm emits masks.mkv for SegMap-only renderers; PR95 is full-RGB.)

OPERATOR-ROUTABLE DESIGN DECISIONS:
  - If the trainer pipeline ever emits a PyTorch state_dict bundle that ALSO
    contains the trained latents tensor, pass `--latents-from-pt` to read
    latents from a `latents` key in the .pt instead of round-tripping from
    the source archive zip. The canonical contract still requires the source
    archive zip for meta (n_pairs, latent_dim, base_channels, eval_size)
    until a sister extension teaches the .pt to carry meta as well.

Sister of Catalog #146 (inflate runtime contract) + Catalog #205 (canonical
select_inflate_device) + Catalog #295 (submission inflate self-containment) +
Catalog #220 (substrate L1+ operational mechanism) + Catalog #272
(distinguishing-feature integration contract).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MODEL = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto"
    / "source/submissions/hnerv_muon"
)
CANONICAL_VENDOR_FILES = ("src/model.py", "src/codec.py")
PR95_MLX_PACKAGE_SCHEMA = "pr95_mlx_pytorch_state_dict_to_contest_archive.v1"

FALSE_AUTHORITY: dict[str, Any] = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _load_pytorch_state_dict(pt_path: Path) -> dict[str, Any]:
    import torch

    if not pt_path.is_file():
        raise FileNotFoundError(f"PyTorch state_dict .pt not found: {pt_path}")
    sd = torch.load(pt_path, weights_only=False, map_location="cpu")
    if not isinstance(sd, dict):
        raise ValueError(
            f"PyTorch state_dict .pt must be a dict, got {type(sd).__name__}"
        )
    return sd


def _resolve_latents_and_meta(
    *,
    source_archive_zip: Path,
    latents_from_pt: bool,
    pt_state_dict: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Resolve (latents, meta, source_custody) from source archive (default) or .pt."""

    from tac.local_acceleration.pr95_hnerv_mlx import parse_pr95_public_archive_zip

    packet = parse_pr95_public_archive_zip(source_archive_zip)
    source_custody = packet.custody_manifest()
    if latents_from_pt:
        if "latents" not in pt_state_dict:
            raise KeyError(
                "--latents-from-pt requested but 'latents' key not in PyTorch .pt; "
                "fall back to default (round-trip from source archive)"
            )
        latents = pt_state_dict.pop("latents")
        return latents, dict(packet.meta), source_custody
    return packet.latents, dict(packet.meta), source_custody


def _strip_pt_keys_not_in_decoder(
    pt_state_dict: dict[str, Any],
    *,
    latent_dim: int,
    base_channels: int,
) -> dict[str, Any]:
    """Filter to canonical PR95 decoder keys (drop optimizer/EMA/metadata)."""

    from tac.local_acceleration.pr95_hnerv_mlx import _expected_pr95_state_shapes

    expected_keys = set(
        _expected_pr95_state_shapes(
            latent_dim=latent_dim,
            base_channels=base_channels,
        )
    )
    return {k: v for k, v in pt_state_dict.items() if k in expected_keys}


def _vendor_canonical_pr95_runtime(
    *,
    submission_dir: Path,
    source_submission_root: Path,
) -> dict[str, str]:
    """Vendor canonical PR 95 model.py + codec.py into submission_dir/src/."""

    vendored: dict[str, str] = {}
    src_dir = submission_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    for rel in CANONICAL_VENDOR_FILES:
        src_path = source_submission_root / rel
        dst_path = submission_dir / rel
        if not src_path.is_file():
            raise FileNotFoundError(
                f"canonical PR 95 source file not found: {src_path}"
            )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        vendored[rel] = _sha256_file(dst_path)
    return vendored


def _write_inflate_sh(submission_dir: Path) -> str:
    """Emit canonical 3-arg inflate.sh per Catalog #146 contract."""

    body = (
        "#!/usr/bin/env bash\n"
        "# PR95 MLX-trained byte-closed contest archive: canonical inflate.sh.\n"
        "# Mirrors the canonical PR 95 hnerv_muon inflate.sh signature\n"
        "# ($1 archive_dir, $2 output_dir, $3 file_list) per Catalog #146.\n"
        "set -euo pipefail\n"
        "\n"
        "HERE=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "\n"
        "DATA_DIR=\"$1\"\n"
        "OUTPUT_DIR=\"$2\"\n"
        "FILE_LIST=\"$3\"\n"
        "\n"
        "mkdir -p \"$OUTPUT_DIR\"\n"
        "\n"
        "while IFS= read -r line; do\n"
        "  [ -z \"$line\" ] && continue\n"
        "  BASE=\"${line%.*}\"\n"
        "  SRC=\"${DATA_DIR}/${BASE}.bin\"\n"
        "  DST=\"${OUTPUT_DIR}/${BASE}.raw\"\n"
        "\n"
        "  [ ! -f \"$SRC\" ] && echo \"ERROR: ${SRC} not found\" >&2 && exit 1\n"
        "\n"
        "  printf \"Inflating %s ... \" \"$line\"\n"
        "  \"${PYTHON:-python3}\" \"$HERE/inflate.py\" \"$SRC\" \"$DST\"\n"
        "done < \"$FILE_LIST\"\n"
    )
    path = submission_dir / "inflate.sh"
    path.write_text(body)
    path.chmod(0o755)
    return _sha256_bytes(body.encode("utf-8"))


def _write_inflate_py(submission_dir: Path) -> str:
    """Emit canonical inflate.py per Catalog #205 select_inflate_device + #295."""

    body = '''#!/usr/bin/env python
"""PR95 MLX-trained byte-closed contest archive inflate.py.

Reads archive bytes (single member layout: meta_blob | decoder_blob |
latents_blob per the canonical PR 95 `build_archive` grammar), runs the
HNeRVDecoder forward, bicubic-upsamples to camera resolution, rounds to uint8,
and writes the contiguous (N, H, W, 3) RGB frame stream to <dst>.

Per Catalog #205 self-protection: honors `PACT_INFLATE_DEVICE`
(auto/cpu/cuda); refuses `mps`. Mirrors canonical
`tac.substrates._shared.inflate_runtime.select_inflate_device` body byte-for-
byte (modulo the `torch.device` return type wrap).

Per Catalog #295 + HNeRV parity discipline lesson 9: the runtime tree is
fully self-contained; `src/model.py` + `src/codec.py` are vendored alongside
this file and imported through a PYTHONPATH-shim that points at the submission-
local `src/` directory.
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from codec import parse_archive  # type: ignore  # noqa: E402
from model import HNeRVDecoder  # type: ignore  # noqa: E402

CAMERA_H, CAMERA_W = 874, 1164


def select_inflate_device() -> torch.device:
    """Honor PACT_INFLATE_DEVICE (auto/cpu/cuda); MPS is forbidden.

    Per Catalog #205 self-protection. Mirrors canonical
    `tac.substrates._shared.inflate_runtime.select_inflate_device` body
    byte-for-byte (modulo the `torch.device` return-type wrap).
    """
    value = (os.environ.get("PACT_INFLATE_DEVICE") or "auto").strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PACT_INFLATE_DEVICE=cuda but torch.cuda is not available"
            )
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    raise RuntimeError(
        f"unsupported PACT_INFLATE_DEVICE={value!r}; expected auto/cpu/cuda"
    )


def inflate(src_bin: str, dst_raw: str) -> int:
    with open(src_bin, "rb") as fin:
        archive_bytes = fin.read()
    decoder_sd, latents, meta = parse_archive(archive_bytes)

    device = select_inflate_device()
    decoder = HNeRVDecoder(
        latent_dim=int(meta["latent_dim"]),
        base_channels=int(meta["base_channels"]),
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    latents = latents.to(device)
    n_pairs = int(meta["n_pairs"])
    eval_h, eval_w = tuple(meta["eval_size"])

    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            batch = j - i
            decoded = decoder(latents[i:j])
            flat = decoded.reshape(batch * 2, 3, eval_h, eval_w)
            up = F.interpolate(
                flat,
                size=(CAMERA_H, CAMERA_W),
                mode="bicubic",
                align_corners=False,
            )
            frames = (
                up.clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            fout.write(frames.tobytes())
            n += batch * 2
    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
'''
    path = submission_dir / "inflate.py"
    path.write_text(body)
    path.chmod(0o755)
    return _sha256_bytes(body.encode("utf-8"))


def _write_readme(
    *,
    submission_dir: Path,
    archive_zip_sha256: str,
    archive_zip_bytes: int,
    decoder_sd_tensor_count: int,
    latent_shape: list[int],
    source_archive_sha256: str,
    pt_sha256: str,
    upstream_pr_number: int = 95,
) -> str:
    body = (
        f"# PR95 MLX-trained byte-closed contest archive\n"
        f"\n"
        f"Packaged by `tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py`\n"
        f"(canonical loop closure cascade #2 per the PR95 MLX cascade plan).\n"
        f"\n"
        f"## Custody\n"
        f"\n"
        f"| Field | Value |\n"
        f"| --- | --- |\n"
        f"| `archive_sha256` | `{archive_zip_sha256}` |\n"
        f"| `archive_size_bytes` | `{archive_zip_bytes:,}` |\n"
        f"| `archive_member_count` | 1 (member name: `0.bin`) |\n"
        f"| `decoder_state_dict_tensor_count` | {decoder_sd_tensor_count} |\n"
        f"| `latent_shape` | `{latent_shape}` |\n"
        f"| `source_archive_sha256` | `{source_archive_sha256}` |\n"
        f"| `pytorch_state_dict_sha256` | `{pt_sha256}` |\n"
        f"| `upstream_pr_number` | {upstream_pr_number} |\n"
        f"\n"
        f"## Inflate\n"
        f"\n"
        f"```bash\n"
        f"./inflate.sh <archive_dir> <output_dir> <file_list>\n"
        f"```\n"
        f"\n"
        f"The runtime tree is self-contained per Catalog #295 + HNeRV parity\n"
        f"discipline lesson 9 (`src/model.py` + `src/codec.py` vendored\n"
        f"alongside `inflate.py`).\n"
        f"\n"
        f"## Score authority\n"
        f"\n"
        f"This packet is `[macOS-MLX research-signal]` source. The byte-closed\n"
        f"archive itself carries no contest-axis score authority until paired\n"
        f"contest CPU + CUDA auth eval lands per CLAUDE.md \"Submission auth\n"
        f"eval - BOTH CPU AND CUDA\" non-negotiable + Catalog #192/#317/#341\n"
        f"canonical-routing-markers discipline.\n"
        f"\n"
        f"## Cascade NEXT\n"
        f"\n"
        f"Loop closure cascade piece #3 = full-frame inflate parity test\n"
        f"(MLX-trained forward vs PyTorch byte-closed archive inflate, on the\n"
        f"contest video, byte-for-byte at the rendered uint8 RGB output).\n"
    )
    path = submission_dir / "README.md"
    path.write_text(body)
    return _sha256_bytes(body.encode("utf-8"))


def _write_archive_manifest(
    *,
    submission_dir: Path,
    archive_zip_path: Path,
    archive_zip_sha256: str,
    archive_zip_bytes: int,
    member_name: str,
    member_bytes: int,
    member_sha256: str,
) -> Path:
    with zipfile.ZipFile(archive_zip_path) as zf:
        info = zf.getinfo(member_name)
    manifest = {
        "archive_path": archive_zip_path.relative_to(REPO_ROOT).as_posix()
        if archive_zip_path.is_absolute() and REPO_ROOT in archive_zip_path.parents
        else archive_zip_path.as_posix(),
        "archive_sha256": archive_zip_sha256,
        "archive_size_bytes": archive_zip_bytes,
        "members": [
            {
                "name": member_name,
                "file_size": int(info.file_size),
                "compress_size": int(info.compress_size),
                "compress_type": int(info.compress_type),
                "crc": int(info.CRC),
                "sha256": member_sha256,
            }
        ],
    }
    path = submission_dir / "archive_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def package_pytorch_state_dict_to_contest_archive(
    *,
    input_pt: Path,
    source_archive_zip: Path,
    output_submission_dir: Path,
    source_submission_root: Path = DEFAULT_SOURCE_MODEL,
    latents_from_pt: bool = False,
    overwrite: bool = True,
    report_out: Path | None = None,
) -> dict[str, Any]:
    """Package PyTorch state_dict into canonical PR 95 byte-closed contest archive.

    Heavy lifting delegated to
    ``tac.local_acceleration.pr95_hnerv_mlx.write_pr95_public_archive_zip``.
    """

    from tac.local_acceleration.pr95_hnerv_mlx import write_pr95_public_archive_zip

    input_pt = Path(input_pt)
    source_archive_zip = Path(source_archive_zip)
    output_submission_dir = Path(output_submission_dir)
    source_submission_root = Path(source_submission_root)

    if output_submission_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"output submission dir exists and --no-overwrite was passed: "
                f"{output_submission_dir}"
            )
        shutil.rmtree(output_submission_dir)
    output_submission_dir.mkdir(parents=True, exist_ok=True)

    pt_state_dict = _load_pytorch_state_dict(input_pt)
    pt_sha256 = _sha256_file(input_pt)
    pt_bytes = input_pt.stat().st_size

    latents, meta, source_custody = _resolve_latents_and_meta(
        source_archive_zip=source_archive_zip,
        latents_from_pt=latents_from_pt,
        pt_state_dict=pt_state_dict,
    )
    decoder_sd = _strip_pt_keys_not_in_decoder(
        pt_state_dict,
        latent_dim=int(meta["latent_dim"]),
        base_channels=int(meta["base_channels"]),
    )

    archive_zip_path = output_submission_dir / "archive.zip"
    write_report = write_pr95_public_archive_zip(
        state_dict=decoder_sd,
        latents=latents,
        meta=meta,
        output_zip_path=archive_zip_path,
        member_name="0.bin",
    )

    vendored = _vendor_canonical_pr95_runtime(
        submission_dir=output_submission_dir,
        source_submission_root=source_submission_root,
    )
    inflate_sh_sha = _write_inflate_sh(output_submission_dir)
    inflate_py_sha = _write_inflate_py(output_submission_dir)
    readme_sha = _write_readme(
        submission_dir=output_submission_dir,
        archive_zip_sha256=write_report["archive_zip_sha256"],
        archive_zip_bytes=int(write_report["archive_zip_bytes"]),
        decoder_sd_tensor_count=len(decoder_sd),
        latent_shape=[int(d) for d in latents.shape]
        if hasattr(latents, "shape")
        else [],
        source_archive_sha256=source_custody["archive_zip_sha256"],
        pt_sha256=pt_sha256,
    )
    manifest_path = _write_archive_manifest(
        submission_dir=output_submission_dir,
        archive_zip_path=archive_zip_path,
        archive_zip_sha256=write_report["archive_zip_sha256"],
        archive_zip_bytes=int(write_report["archive_zip_bytes"]),
        member_name=write_report["member_name"],
        member_bytes=int(write_report["member_bytes"]),
        member_sha256=write_report["member_sha256"],
    )

    runtime_files = {
        "inflate.sh": inflate_sh_sha,
        "inflate.py": inflate_py_sha,
        "README.md": readme_sha,
        **{f"vendored_{rel.replace('/', '_')}": sha for rel, sha in vendored.items()},
    }

    report = {
        "schema_version": PR95_MLX_PACKAGE_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "tool": "tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py",
        "input_pt_path": input_pt.as_posix(),
        "input_pt_sha256": pt_sha256,
        "input_pt_bytes": pt_bytes,
        "source_archive_zip_path": source_archive_zip.as_posix(),
        "source_archive_zip_sha256": source_custody["archive_zip_sha256"],
        "source_archive_member_sha256": source_custody["member_sha256"],
        "source_submission_root": source_submission_root.as_posix(),
        "output_submission_dir": output_submission_dir.as_posix(),
        "archive_zip_path": str(archive_zip_path),
        "archive_zip_sha256": write_report["archive_zip_sha256"],
        "archive_zip_bytes": int(write_report["archive_zip_bytes"]),
        "archive_member_name": write_report["member_name"],
        "archive_member_bytes": int(write_report["member_bytes"]),
        "archive_member_sha256": write_report["member_sha256"],
        "archive_member_compress_type": int(write_report["member_compress_type"]),
        "parsed_meta_after_roundtrip": write_report["parsed_meta"],
        "parsed_latent_shape_after_roundtrip": write_report["parsed_latent_shape"],
        "parsed_state_dict_tensor_count_after_roundtrip": write_report[
            "parsed_state_dict_tensor_count"
        ],
        "decoder_state_dict_tensor_count": len(decoder_sd),
        "latent_shape": [int(d) for d in latents.shape]
        if hasattr(latents, "shape")
        else [],
        "runtime_files_emitted": runtime_files,
        "archive_manifest_path": str(manifest_path),
        "axis_tag": "[macOS-CPU advisory]",
        "evidence_grade": "predicted",
        "canonical_provenance": {
            "source_pr": 95,
            "submission": "hnerv_muon",
            "canonical_helper": "tac.local_acceleration.pr95_hnerv_mlx.write_pr95_public_archive_zip",
            "canonical_equation_id": "pr95_mlx_pytorch_to_byte_closed_contest_archive_pipeline_v1",
        },
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": [
                "pr95_mlx_byte_closed_archive_is_packaged_but_not_runtime_consumed",
                "requires_full_frame_inflate_parity_before_runtime_consumption_claim",
                "requires_paired_contest_cpu_and_cuda_auth_eval_before_score_claim",
            ],
        },
        **FALSE_AUTHORITY,
    }

    if report_out is not None:
        report_out = Path(report_out)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-pt",
        type=Path,
        required=True,
        help="Path to PyTorch state_dict .pt (output of "
        "tools/export_pr95_mlx_to_pytorch_state_dict.py).",
    )
    parser.add_argument(
        "--source-archive-zip",
        type=Path,
        required=True,
        help="Path to source PR 95 public archive .zip (canonical source of "
        "the latents tensor + meta).",
    )
    parser.add_argument(
        "--output-submission-dir",
        type=Path,
        required=True,
        help="Output submission dir to write archive.zip + inflate.sh + "
        "inflate.py + src/ + README.md + archive_manifest.json.",
    )
    parser.add_argument(
        "--source-submission-root",
        type=Path,
        default=DEFAULT_SOURCE_MODEL,
        help="Root of the canonical PR 95 hnerv_muon submission (used to "
        "vendor src/model.py + src/codec.py).",
    )
    parser.add_argument(
        "--latents-from-pt",
        action="store_true",
        help="Read latents tensor from PT 'latents' key instead of "
        "round-tripping from source archive zip (rare; requires PT bundle to "
        "include trained latents).",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        help="Optional path to write a canonical packaging report JSON.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Refuse to overwrite an existing output submission dir.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for any RNG-touching paths (currently no RNG used).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = package_pytorch_state_dict_to_contest_archive(
        input_pt=args.input_pt,
        source_archive_zip=args.source_archive_zip,
        output_submission_dir=args.output_submission_dir,
        source_submission_root=args.source_submission_root,
        latents_from_pt=args.latents_from_pt,
        overwrite=not args.no_overwrite,
        report_out=args.report_out,
    )
    print(
        f"[pr95-mlx-package] archive_zip={report['archive_zip_path']} "
        f"bytes={report['archive_zip_bytes']} "
        f"sha256={report['archive_zip_sha256'][:16]}..."
    )
    print(
        f"[pr95-mlx-package] member={report['archive_member_name']} "
        f"bytes={report['archive_member_bytes']} "
        f"sha256={report['archive_member_sha256'][:16]}..."
    )
    print(
        f"[pr95-mlx-package] runtime_files={list(report['runtime_files_emitted'].keys())}"
    )
    if args.report_out is not None:
        print(f"[pr95-mlx-package] report={args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
