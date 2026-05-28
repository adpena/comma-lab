#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Repack a PSV3 archive with decoder-side quantization.

This is an encoder/compression-side tool. It rewrites only the archive bytes
and runtime packet, emits a deterministic local replay proof, and remains
fail-closed for score authority until paired contest CPU/CUDA eval consumes
the resulting byte-closed archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SCHEMA_VERSION = "pact_nerv_selector_v3_decoder_quant_repack.v1"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_psv3_bytes(path: Path) -> tuple[bytes, str]:
    from tac.substrates.pact_nerv_selector_v3.archive import PSV3_MAGIC

    raw = path.read_bytes()
    if raw[:4] == PSV3_MAGIC:
        return raw, "raw_0_bin"
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is neither raw PSV3 bytes nor a ZIP archive")
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if "0.bin" not in names:
            raise ValueError(f"{path} is missing ZIP member 0.bin; members={names}")
        data = zf.read("0.bin")
    if data[:4] != PSV3_MAGIC:
        raise ValueError(f"ZIP member 0.bin is not PSV3: magic={data[:4]!r}")
    return data, "zip_member_0_bin"


def _cfg_from_archive(arc: Any) -> Any:
    from tac.substrates.pact_nerv_selector_v3.architecture import (
        PactNervSelectorV3Config,
    )

    meta = arc.meta
    return PactNervSelectorV3Config(
        latent_dim=int(arc.latents.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        sin_frequency=float(meta["sin_frequency"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
        num_pairs=int(arc.latents.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
        selector_palette_size=int(arc.palette_size),
        rice_golomb_k=int(meta.get("rice_golomb_k", 2)),
    )


def _build_model_from_archive(arc: Any) -> Any:
    import torch

    from tac.substrates.pact_nerv_selector_v3.architecture import (
        PactNervSelectorV3Substrate,
    )

    cfg = _cfg_from_archive(arc)
    model = PactNervSelectorV3Substrate(cfg).eval()
    load_result = model.load_state_dict(arc.decoder_state_dict, strict=False)
    unexpected = set(load_result.unexpected_keys)
    missing = set(load_result.missing_keys) - {"latents", "selectors"}
    if unexpected or missing:
        raise RuntimeError(
            "PSV3 decoder load mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    with torch.no_grad():
        model.latents.copy_(arc.latents.to(dtype=model.latents.dtype))
    return model


def _measure_decoder_drift(source_arc: Any, candidate_arc: Any, n_pairs: int) -> dict[str, Any]:
    import torch

    src_model = _build_model_from_archive(source_arc)
    cand_model = _build_model_from_archive(candidate_arc)
    n = max(1, min(int(n_pairs), int(source_arc.latents.shape[0])))
    idx = torch.arange(n, dtype=torch.long)
    t0 = time.perf_counter()
    with torch.no_grad():
        s0, s1 = src_model(idx)
        c0, c1 = cand_model(idx)
    render_seconds = time.perf_counter() - t0
    src = torch.stack((s0, s1), dim=1)
    cand = torch.stack((c0, c1), dim=1)
    drift = (src - cand).abs()
    return {
        "n_pairs_measured": n,
        "frame_shape": list(src.shape),
        "max_abs_drift_01": float(drift.max()),
        "mean_abs_drift_01": float(drift.mean()),
        "render_seconds_cpu": float(render_seconds),
        "decoder_output_space": "sigmoid_0_to_1",
    }


def repack_pact_nerv_selector_v3_decoder_quant(
    *,
    archive: Path,
    output_dir: Path,
    decoder_quantization: str,
    n_proof_pairs: int,
    candidate_label: str,
) -> dict[str, Any]:
    from tac.substrates._shared.pact_nerv_full_main import (
        build_archive_zip,
        write_contest_runtime,
    )
    from tac.substrates.pact_nerv_selector_v3.archive import (
        DECODER_QUANTIZATION_KINDS,
        pack_archive,
        parse_archive,
    )

    if decoder_quantization not in DECODER_QUANTIZATION_KINDS:
        raise ValueError(
            f"unsupported decoder_quantization={decoder_quantization!r}; "
            f"expected one of {sorted(DECODER_QUANTIZATION_KINDS)}"
        )
    source_bytes, source_kind = _read_psv3_bytes(archive)
    source_arc = parse_archive(source_bytes)
    meta = dict(source_arc.meta)
    meta.pop("decoder_quantization", None)
    candidate_bytes = pack_archive(
        source_arc.decoder_state_dict,
        source_arc.latents,
        source_arc.selector_bytes,
        meta,
        palette_size=int(source_arc.palette_size),
        decoder_quantization=decoder_quantization,
    )
    candidate_arc = parse_archive(candidate_bytes)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_0bin = output_dir / "0.bin"
    candidate_0bin.write_bytes(candidate_bytes)
    submission_dir = output_dir / "submission"
    write_contest_runtime(
        submission_dir,
        substrate_pkg_name="pact_nerv_selector_v3",
        repo_root=REPO_ROOT,
    )
    (submission_dir / "0.bin").write_bytes(candidate_bytes)
    archive_zip = output_dir / "archive.zip"
    build_archive_zip(archive_zip, bin_bytes=candidate_bytes, submission_dir=submission_dir)

    drift = _measure_decoder_drift(source_arc, candidate_arc, n_proof_pairs)
    source_size = len(source_bytes)
    candidate_size = len(candidate_bytes)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "tool": "tools/repack_pact_nerv_selector_v3_decoder_quant.py",
        "candidate_label": candidate_label,
        "source_archive": str(archive),
        "source_archive_kind": source_kind,
        "source_0bin_sha256": _sha256_bytes(source_bytes),
        "source_0bin_bytes": source_size,
        "candidate_0bin_path": str(candidate_0bin),
        "candidate_0bin_sha256": _sha256_bytes(candidate_bytes),
        "candidate_0bin_bytes": candidate_size,
        "candidate_archive_zip_path": str(archive_zip),
        "candidate_archive_zip_sha256": _sha256_file(archive_zip),
        "candidate_archive_zip_bytes": archive_zip.stat().st_size,
        "decoder_quantization": decoder_quantization,
        "rate_delta_0bin_bytes": candidate_size - source_size,
        "rate_delta_0bin_fraction": (
            (candidate_size - source_size) / source_size if source_size else 0.0
        ),
        "rate_win": candidate_size < source_size,
        "local_decoder_drift": drift,
        "axis_tag": "[macOS-CPU archive-proof]",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": [
            "local_archive_repack_is_not_contest_score_authority",
            "requires_paired_contest_cpu_plus_cuda_eval_before_score_claim",
            "requires_dispatch_claim_and_runtime_custody_handoff_before_exact_eval",
        ],
    }
    manifest_path = output_dir / "decoder_quant_repack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repack a PSV3 archive with decoder-side quantization."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--decoder-quantization",
        default="int8_per_channel_brotli_q11",
        choices=[
            "fp16_brotli_q9",
            "fp16_brotli_q11",
            "int8_per_channel_brotli_q11",
        ],
    )
    parser.add_argument("--n-proof-pairs", type=int, default=1)
    parser.add_argument(
        "--candidate-label",
        default="pact_nerv_selector_v3_decoder_quant_repack",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = repack_pact_nerv_selector_v3_decoder_quant(
            archive=args.archive,
            output_dir=args.output_dir,
            decoder_quantization=args.decoder_quantization,
            n_proof_pairs=args.n_proof_pairs,
            candidate_label=args.candidate_label,
        )
    except Exception as exc:
        print(f"[psv3-decoder-quant] ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
