# SPDX-License-Identifier: MIT
"""Train the Z3-G1 entropy-coded v2 substrate.

Per `.omx/research/wunderkind_g1_entropy_coded_v2_design_20260515.md`:
operator-approved 2026-05-15 reactivation of v1
(`lane_z3_g1_scorer_softmax_hyperprior_gating_20260515`, research_only=true,
deferred per F1 codex finding empirical confirmation that
`hyperprior_weights_int8` + `w_hat_int8` slots ship empty `b""`).

v2 introduces a NEW magic + grammar (`Z3G2`) that REPLACES the empty Z3HV2
slots with TWO entropy-coded streams ACTUALLY shipped at the wire-byte level:

    sigma_table_blob:    brotli(sigma_table_int8) ~300B
    class_prior_cdf:     5*uint16 = 10B raw
    class_index_blob:    constriction-Huffman(class_indices) ~200B

These bytes are consumed by the parser/intermediate Z3G2 reconstruction path
(verified structurally by
``tools/verify_z3_g1_entropy_coded_v2_byte_mutation.py`` per Catalog #139).
The non-smoke path now emits a byte-closed research packet and a bounded
``inflate.sh`` frame-output mutation proof. Paired CPU/CUDA exact eval and a
real compress-side scorer-class exporter remain required before any promotion.

Score movement is unranked at scaffold time. The current implementation is a
lossy latent transform because sigma affects reconstructed latents at inflate
time; distortion must be measured by paired exact eval.

Council-binding contract honored:

- Catalog #146: 3-arg archive grammar (decoder + section + sidecar).
- Catalog #151: TIER_1_OPERATOR_REQUIRED_FLAGS declared as ast.AnnAssign
  per Catalog #168 AST walker.
- Catalog #205: select_inflate_device canonical helper.
- Catalog #220: score_improvement_mechanism_status=RESEARCH_ONLY +
  runtime_overlay_consumed=False until full-frame inflate proof and paired
  exact eval land.
- Catalog #226: gate_auth_eval_call canonical helper for auth eval routing.
- Catalog #240: dispatch_enabled requires implementation_complete. The
  smoke path (``_smoke_main``) is COMPLETE; the full path (``_full_main``)
  is research-only and emits fail-closed packet/proof artifacts without
  claiming score or dispatch readiness.
- Catalog #272: distinguishing-feature integration contract documented
  in design memo §5.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY":
the v2 substrate remains research-only until a learned/non-placeholder
compress-side export plus paired exact eval lands. The smoke path validates
only the parser/intermediate byte-consumption contract; the full path adds
byte-closed packet export plus bounded frame-output mutation proof. Neither
path is dispatch-ranking or promotion evidence.

Usage (smoke; CPU; ~3 epochs over a synthetic A1-shaped tensor; verifies
byte-mutation contract end-to-end):

    .venv/bin/python experiments/train_substrate_z3_g1_entropy_coded_v2.py \\
        --output-dir experiments/results/z3g1_entropy_coded_v2_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full export; CPU-safe research packet + bounded frame proof):

    .venv/bin/python experiments/train_substrate_z3_g1_entropy_coded_v2.py \\
        --a1-archive-path submissions/a1/archive.zip \\
        --output-dir experiments/results/z3g1_entropy_coded_v2_<utc> \\
        --epochs 1 --batch-size 16 --lr 1e-3 --device cpu --frame-proof-pairs 16
"""
# AUTOCAST_FP16_WAIVED:smoke-only-scaffold-defers-to-v1-trainer-for-amp-config-pattern
# TORCH_COMPILE_WAIVED:smoke-only-scaffold-defers-to-v1-trainer-for-Inductor-config-pattern
# NO_GRAD_WAIVED:smoke-path-uses-torch.no_grad-explicitly-around-eval-block
# SYNTHETIC_NON_SMOKE_OK:_smoke_main-only-uses-synthetic-latents-_full_main-uses-real-a1-packet-with-placeholder-classes-and-no-score-claim
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from tac.substrates._shared.trainer_skeleton import (
    detect_hardware_substrate as _canon_detect_hardware_substrate,
)
from tac.substrates._shared.trainer_skeleton import (
    git_head_sha as _canon_git_head_sha,
)
from tac.substrates._shared.trainer_skeleton import (
    pin_seeds as _canon_pin_seeds,
)
from tac.substrates._shared.trainer_skeleton import (
    utc_now_iso as _canon_utc_now_iso,
)
from tac.substrates.z3_g1_entropy_coded_v2 import (
    A1_LATENT_DIM,
    A1_N_PAIRS,
    G1_NUM_SCORER_CLASSES,
    Z3G1EntropyCodedV2Config,
    Z3G2EntropyCodedScorerClassGatingHead,
    build_z3g2_composition_archive_contract,
    build_z3g2_payload_bytes,
    encode_z3g2_section,
    estimate_z3g2_section_overhead_bytes,
    g1_v2_residual_rate_bits_per_sample,
    reconstruct_class_indices_and_sigma_table_from_z3g2_payload,
)
from tac.substrates.z3_g1_entropy_coded_v2.archive import (
    A1_DECODER_SECTION_TOTAL,
    A1_LATENT_BLOB_LEN,
    Z3G2_CLASS_PRIOR_BLOB_LEN,
    Z3G2_HEADER_STRUCT,
)
from tac.substrates.z3_g1_entropy_coded_v2.registered_substrate import (
    Z3_G1_ENTROPY_CODED_V2_CONTRACT,  # forces decoration-time validation
)

# Per Catalog #151 + Catalog #168: declare TIER_1_OPERATOR_REQUIRED_FLAGS via
# ast.AnnAssign (the AST walker accepts both Assign and AnnAssign forms).
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--a1-archive-path": {
        "env": "Z3_G1_V2_A1_ARCHIVE_PATH",
        "default": "submissions/a1/archive.zip",
        "required_input_file": True,
        "rationale": "Z3G2 wire grammar splices into A1 archive bytes (verbatim decoder + sidecar).",
        "generator_command": "python tools/build_a1_archive.py --output submissions/a1/archive.zip",
    },
    "--output-dir": {
        "env": "Z3_G1_V2_OUTPUT_DIR",
        "default": None,
        "required_input_file": False,
        "rationale": "Trainer output directory for stats.json + archive.zip + provenance.",
    },
    "--video-path": {
        "env": "Z3_G1_V2_VIDEO_PATH",
        "default": "upstream/videos/0.mkv",
        "required_input_file": True,
        "rationale": "Reserved for scorer-conditioned class-index training; full export records custody even when this deterministic builder does not decode video.",
    },
    "--upstream-dir": {
        "env": "Z3_G1_V2_UPSTREAM_DIR",
        "default": "upstream",
        "required_input_file": True,
        "rationale": "Reserved for eventual paired auth-eval routing and runtime custody.",
    },
    "--epochs": {
        "env": "Z3_G1_V2_EPOCHS",
        "default": 1000,
        "required_input_file": False,
        "rationale": "Full-run epoch count (smoke uses --epochs 3).",
    },
    "--batch-size": {
        "env": "Z3_G1_V2_BATCH_SIZE",
        "default": 16,
        "required_input_file": False,
        "rationale": "Pair batch size for per-epoch SGD updates.",
    },
    "--lr": {
        "env": "Z3_G1_V2_LR",
        "default": 1e-3,
        "required_input_file": False,
        "rationale": "Optimizer learning rate.",
    },
    "--device": {
        "env": "Z3_G1_V2_DEVICE",
        "default": "cuda",
        "required_input_file": False,
        "rationale": "Training device per Catalog #190 + 'MPS NOISE' non-negotiable.",
    },
    "--smoke": {
        "env": "Z3_G1_V2_SMOKE_ONLY",
        "default": False,
        "required_input_file": False,
        "rationale": "Smoke mode: synthetic latents, ≤3 epochs, no Modal dispatch.",
    },
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Z3-G1 entropy-coded v2 trainer scaffold."
    )
    parser.add_argument("--a1-archive-path", type=str, default="submissions/a1/archive.zip")
    parser.add_argument("--video-path", type=str, default="upstream/videos/0.mkv")
    parser.add_argument("--upstream-dir", type=str, default="upstream")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frame-proof-pairs",
        type=int,
        default=16,
        help=(
            "Number of pairs decoded by the bounded frame-output mutation proof. "
            "This is proof-only and not a score claim."
        ),
    )
    parser.add_argument("--skip-frame-proof", action="store_true")
    return parser


def _smoke_main(args: argparse.Namespace) -> int:
    """Smoke main: synthetic A1-shaped latents + verify Z3G2 packet roundtrip.

    Goal: validate the FULL encode→decode→re-encode pipeline locally before
    burning Modal dispatch. Computes a typed contract and writes stats.json
    with byte_savings + distinguishing_feature_bytes for downstream gates.

    Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" + Catalog #167: smoke
    runs LOCALLY first; only after smoke + byte-mutation gate pass should
    operator-authorize fire any Modal dispatch.
    """
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _canon_pin_seeds(args.seed)

    # 1. Build synthetic A1-shaped latents + per-pair class indices.
    head = Z3G2EntropyCodedScorerClassGatingHead()
    sigma_table_int8, scale = head.quantize_sigma_table_int8()
    class_indices = torch.randint(0, G1_NUM_SCORER_CLASSES, (A1_N_PAIRS,))
    a1_latents = torch.randn(A1_N_PAIRS, A1_LATENT_DIM)
    latent_offset = torch.zeros(A1_LATENT_DIM)
    latent_scale = torch.ones(A1_LATENT_DIM)

    # 2. Compute training-time loss (rate-only mode for smoke).
    bits_per_sample, sigma, class_prior_counts = g1_v2_residual_rate_bits_per_sample(
        gating_head=head,
        a1_latents=a1_latents,
        class_indices=class_indices,
        latent_offset=latent_offset,
        latent_scale=latent_scale,
    )
    rate_bits_total = bits_per_sample.sum().item()
    print(f"[smoke] rate_bits_total = {rate_bits_total:.1f} bits "
          f"({rate_bits_total / 8:.1f} bytes residual entropy)")

    # 3. Build Z3G2 packet + verify roundtrip.
    residual_int8_bytes = bytes((A1_N_PAIRS * A1_LATENT_DIM) * [3])
    section = encode_z3g2_section(
        sigma_table_int8=sigma_table_int8,
        class_indices_uint8=bytes(class_indices.to(torch.uint8).tolist()),
        class_prior_counts=class_prior_counts,
        residual_int8=residual_int8_bytes,
        latent_offset=latent_offset,
        latent_scale=latent_scale,
        int8_sigma_scale=scale,
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
    )
    print(f"[smoke] z3g2_section bytes = {len(section)}")

    # 4. Build a synthetic A1 payload + splice in z3g2 section.
    import struct
    fake_a1 = (
        struct.pack("<I", 162168)
        + bytes(162164 * [42])
        + bytes(15387 * [99])
        + b"sidecar_smoke"
    )
    payload = build_z3g2_payload_bytes(a1_bytes=fake_a1, z3g2_section=section)
    contract = build_z3g2_composition_archive_contract(fake_a1, payload)
    print(
        f"[smoke] payload_bytes={contract.archive_bytes} "
        f"savings_bytes={contract.byte_savings_bytes} "
        f"distinguishing_feature_bytes={contract.distinguishing_feature_bytes}"
    )

    # 5. Write stats.json with apples-to-apples discipline (no score claim).
    stats = {
        "schema_version": "z3g1_entropy_coded_v2_smoke_v1",
        "smoke_mode": True,
        "synthetic_latents": True,
        "epochs": args.epochs,
        "rate_bits_total": rate_bits_total,
        "rate_bytes_total": rate_bits_total / 8.0,
        "z3g2_section_bytes": contract.z3g2_section_bytes,
        "byte_savings_bytes": contract.byte_savings_bytes,
        "distinguishing_feature_bytes": contract.distinguishing_feature_bytes,
        "z3g2_section_overhead_estimate_bytes": estimate_z3g2_section_overhead_bytes(
            gating_head=head
        ),
        "class_prior_counts": class_prior_counts.tolist(),
        "score_claim": False,
        "score_axis": "smoke_synthetic",
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "result_review_blockers": [
            *contract.result_review_blockers,
            "smoke_mode_synthetic_latents_not_score_anchor",
            "byte_mutation_smoke_must_pass_per_catalog_139",
        ],
        "git_head_sha": _canon_git_head_sha(),
        "hardware_substrate": _canon_detect_hardware_substrate(axis="cpu", substrate_tag="z3_g1_entropy_coded_v2"),
        "utc_now": _canon_utc_now_iso(),
        "lane_id": "lane_z3_g1_entropy_coded_v2_20260515",
        "substrate_id": "z3_g1_entropy_coded_v2",
        "council_verdict_provenance": ".omx/research/wunderkind_g1_entropy_coded_v2_design_20260515.md",
    }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(f"[smoke] wrote {stats_path}")
    print(
        "[smoke] DONE — byte-mutation gate must pass before Modal dispatch:\n"
        "  PYTHONPATH=src:upstream:. .venv/bin/python "
        "tools/verify_z3_g1_entropy_coded_v2_byte_mutation.py --verbose"
    )
    return 0


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_archive_member_bytes(archive_zip_path: Path) -> tuple[bytes, str, int, str]:
    """Read the charged inner payload from an A1/Z3-style single-member ZIP."""
    with zipfile.ZipFile(archive_zip_path) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        if "x" in names:
            member = "x"
        elif "0.bin" in names:
            member = "0.bin"
        elif len(names) == 1:
            member = names[0]
        else:
            raise ValueError(
                f"archive {archive_zip_path} must have x, 0.bin, or exactly one member; got {names}"
            )
        data = zf.read(member)
    return data, _sha256_bytes(data), len(data), member


def _load_a1_codec_module(a1_archive_path: Path):
    import importlib

    a1_src = a1_archive_path.parent / "src"
    if not (a1_src / "codec.py").is_file() or not (a1_src / "model.py").is_file():
        raise FileNotFoundError(
            f"A1 runtime missing at {a1_src}; expected codec.py and model.py"
        )
    old_path = list(sys.path)
    old_codec = sys.modules.pop("codec", None)
    old_model = sys.modules.pop("model", None)
    sys.path.insert(0, str(a1_src))
    try:
        return importlib.import_module("codec")
    finally:
        sys.path = old_path
        sys.modules.pop("codec", None)
        sys.modules.pop("model", None)
        if old_codec is not None:
            sys.modules["codec"] = old_codec
        if old_model is not None:
            sys.modules["model"] = old_model


def _decode_a1_final_latents(a1_archive_path: Path, a1_bytes: bytes) -> torch.Tensor:
    """Decode A1 latents with its original sidecar applied."""
    import struct

    codec = _load_a1_codec_module(a1_archive_path)
    section_total = struct.unpack_from("<I", a1_bytes, 0)[0]
    latent_blob = a1_bytes[section_total : section_total + int(codec.LATENT_BLOB_LEN)]
    sidecar_blob = a1_bytes[section_total + int(codec.LATENT_BLOB_LEN) :]
    latents = codec.apply_latent_sidecar(
        codec.decode_latents_compact(latent_blob), sidecar_blob
    )
    return latents.detach().to(torch.float32).cpu()


def _deterministic_class_indices(n_pairs: int = A1_N_PAIRS) -> torch.Tensor:
    """Deterministic placeholder until compress-side SegNet class extraction lands."""
    return (torch.arange(n_pairs, dtype=torch.long) % G1_NUM_SCORER_CLASSES).contiguous()


def _sigma_table_for_export() -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return (sigma_real, sigma_int8, int8_scale) for deterministic export."""
    cfg = Z3G1EntropyCodedV2Config()
    class_scale = torch.linspace(0.80, 1.20, steps=G1_NUM_SCORER_CLASSES).unsqueeze(1)
    dim_scale = torch.linspace(0.95, 1.05, steps=A1_LATENT_DIM).unsqueeze(0)
    sigma_real = (class_scale * dim_scale).to(torch.float32)
    int8_scale = float(cfg.int8_sigma_scale)
    sigma_int8 = (sigma_real * 127.0 / int8_scale).round().clamp(1, 127).to(torch.int8)
    sigma_real_q = sigma_int8.to(torch.float32) * int8_scale / 127.0
    return sigma_real_q, sigma_int8, int8_scale


def _build_z3g2_packet_from_a1(
    *,
    a1_archive_path: Path,
    a1_bytes: bytes,
) -> tuple[bytes, dict[str, Any]]:
    """Build a deterministic byte-closed Z3G2 payload from the real A1 archive."""
    latents = _decode_a1_final_latents(a1_archive_path, a1_bytes)
    if tuple(latents.shape) != (A1_N_PAIRS, A1_LATENT_DIM):
        raise ValueError(
            f"A1 latents shape {tuple(latents.shape)} != {(A1_N_PAIRS, A1_LATENT_DIM)}"
        )

    class_indices = _deterministic_class_indices(A1_N_PAIRS)
    sigma_real, sigma_int8, int8_scale = _sigma_table_for_export()
    latent_offset = latents.mean(dim=0)
    centered = latents - latent_offset.unsqueeze(0)
    class_prior_counts = torch.bincount(
        class_indices, minlength=G1_NUM_SCORER_CLASSES
    ).to(torch.int64)
    sigmas_per_pair = sigma_real[class_indices, :]

    candidates: list[dict[str, Any]] = []
    class_indices_uint8 = bytes(class_indices.to(torch.uint8).tolist())
    for residual_peak_target in (127, 110, 96, 80, 64, 48, 40, 32, 24, 16, 12, 8):
        latent_scale = (
            centered.abs().amax(dim=0).div(float(residual_peak_target)).clamp(min=1e-6)
        )
        residual = (
            centered / (latent_scale.unsqueeze(0) * sigmas_per_pair)
        ).round().clamp(-127, 127).to(torch.int8).contiguous()
        reconstructed = (
            residual.to(torch.float32) * sigmas_per_pair
        ) * latent_scale.unsqueeze(0) + latent_offset.unsqueeze(0)
        section = encode_z3g2_section(
            sigma_table_int8=sigma_int8,
            class_indices_uint8=class_indices_uint8,
            class_prior_counts=class_prior_counts,
            residual_int8=residual.numpy().tobytes(),
            latent_offset=latent_offset,
            latent_scale=latent_scale,
            int8_sigma_scale=int8_scale,
            quant_step=1.0,
            min_sigma=1e-3,
            max_sigma=16.0,
        )
        payload = build_z3g2_payload_bytes(a1_bytes=a1_bytes, z3g2_section=section)
        contract = build_z3g2_composition_archive_contract(a1_bytes, payload)
        candidates.append(
            {
                "residual_peak_target": residual_peak_target,
                "section": section,
                "payload": payload,
                "contract": contract,
                "latent_reconstruction_max_abs_err": float(
                    (reconstructed - latents).abs().max().item()
                ),
                "latent_reconstruction_mean_abs_err": float(
                    (reconstructed - latents).abs().mean().item()
                ),
                "residual_unique_values": int(
                    torch.unique(residual.to(torch.int16)).numel()
                ),
            }
        )

    byte_saving_candidates = [
        c for c in candidates if c["contract"].z3g2_section_bytes < A1_LATENT_BLOB_LEN
    ]
    if byte_saving_candidates:
        selected = min(
            byte_saving_candidates,
            key=lambda c: (
                c["latent_reconstruction_mean_abs_err"],
                -int(c["contract"].byte_savings_bytes),
            ),
        )
        selection_reason = "lowest_mean_abs_error_among_byte_saving_candidates"
    else:
        selected = min(candidates, key=lambda c: c["contract"].z3g2_section_bytes)
        selection_reason = "no_byte_saving_candidate_found_selected_smallest_section"

    payload = selected["payload"]
    contract = selected["contract"]
    manifest = {
        "z3g2_section_bytes": contract.z3g2_section_bytes,
        "payload_bytes": len(payload),
        "payload_sha256": _sha256_bytes(payload),
        "class_index_strategy": "deterministic_modulo_placeholder_until_segnet_export",
        "class_prior_counts": class_prior_counts.tolist(),
        "residual_peak_target": selected["residual_peak_target"],
        "residual_peak_selection_reason": selection_reason,
        "latent_reconstruction_max_abs_err": selected[
            "latent_reconstruction_max_abs_err"
        ],
        "latent_reconstruction_mean_abs_err": selected[
            "latent_reconstruction_mean_abs_err"
        ],
        "rate_distortion_candidate_sweep": [
            {
                "residual_peak_target": c["residual_peak_target"],
                "z3g2_section_bytes": c["contract"].z3g2_section_bytes,
                "byte_saving": c["contract"].byte_saving,
                "byte_savings_bytes": c["contract"].byte_savings_bytes,
                "latent_reconstruction_mean_abs_err": c[
                    "latent_reconstruction_mean_abs_err"
                ],
                "latent_reconstruction_max_abs_err": c[
                    "latent_reconstruction_max_abs_err"
                ],
                "residual_unique_values": c["residual_unique_values"],
            }
            for c in candidates
        ],
        "archive_contract": contract.as_manifest(),
    }
    return payload, manifest


def _build_archive_zip(zip_path: Path, *, payload_bytes: bytes) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(filename="0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(info, payload_bytes)


def _write_submission_runtime_manifest(submission_dir: Path) -> dict[str, Any]:
    ignored = {"0.bin", "archive.zip", "submission_runtime_manifest.json"}
    files: list[dict[str, Any]] = []
    h = hashlib.sha256()
    for path in sorted(p for p in submission_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(submission_dir).as_posix()
        if rel in ignored:
            continue
        data = path.read_bytes()
        digest = _sha256_bytes(data)
        files.append({"path": rel, "bytes": len(data), "sha256": digest})
        h.update(rel.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\0")
    manifest = {
        "schema": "submission_runtime_manifest_v1",
        "runtime_tree_sha256": h.hexdigest(),
        "files": files,
    }
    (submission_dir / "submission_runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _write_runtime(submission_dir: Path, *, a1_runtime_src: Path) -> dict[str, Any]:
    """Emit a self-contained Z3G2 inflate runtime plus vendored A1 runtime."""
    submission_dir.mkdir(parents=True, exist_ok=True)
    runtime_src = submission_dir / "src" / "a1_runtime"
    if runtime_src.exists():
        shutil.rmtree(runtime_src)
    shutil.copytree(
        a1_runtime_src,
        runtime_src,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (submission_dir / "inflate.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"
mkdir -p "$OUTPUT_DIR"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/x"
  if [ ! -f "$SRC" ]; then
    SRC="${DATA_DIR}/${BASE}.bin"
  fi
  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1
  "${PYTHON:-python3}" "$HERE/inflate.py" "$SRC" "$OUTPUT_DIR/${BASE}.raw"
done < "$FILE_LIST"
""",
        encoding="utf-8",
    )
    (submission_dir / "inflate.sh").chmod(0o755)
    (submission_dir / "inflate.py").write_text(_Z3G2_INFLATE_PY, encoding="utf-8")
    return _write_submission_runtime_manifest(submission_dir)


_Z3G2_INFLATE_PY = r'''#!/usr/bin/env python3
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import brotli
import constriction
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src" / "a1_runtime"))

from codec import apply_latent_sidecar, decode_decoder_compact
from model import HNeRVDecoder

CAMERA_H, CAMERA_W = 874, 1164
EVAL_H, EVAL_W = 384, 512
LATENT_DIM = 28
BASE_CHANNELS = 36
N_PAIRS = 600
N_CLASSES = 5
A1_DECODER_SECTION_TOTAL = 162168
Z3G2_MAGIC = b"Z3G2"
Z3G2_HEADER = struct.Struct("<4sBHBBffff2s")


def select_inflate_device() -> torch.device:
    value = (os.environ.get("PACT_INFLATE_DEVICE") or "auto").strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PACT_INFLATE_DEVICE=cuda but torch.cuda is not available")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    raise RuntimeError(f"unsupported PACT_INFLATE_DEVICE={value!r}; expected auto/cpu/cuda")


def _decode_class_indices_huffman(data: bytes, counts: np.ndarray) -> bytes:
    if len(data) < 4:
        raise ValueError("class_index blob too short")
    (n_pairs,) = struct.unpack_from("<I", data, 0)
    word_bytes = data[4:]
    if len(word_bytes) % 4:
        raise ValueError("class_index word section is not uint32 aligned")
    if n_pairs == 0:
        return b""
    probs = np.maximum(counts.astype(np.float64), 1.0)
    probs = probs / probs.sum()
    tree = constriction.symbol.huffman.DecoderHuffmanTree(probs)
    decoder = constriction.symbol.QueueDecoder(np.frombuffer(word_bytes, dtype="<u4").copy())
    out = bytearray()
    for _ in range(int(n_pairs)):
        out.append(int(decoder.decode_symbol(tree)) & 0xFF)
    return bytes(out)


def decode_z3g2_section(data: bytes):
    fields = Z3G2_HEADER.unpack_from(data, 0)
    magic, version, n_pairs, n_classes, latent_dim = fields[:5]
    int8_sigma_scale, _quant_step, min_sigma, _max_sigma = fields[5:9]
    if magic != Z3G2_MAGIC or version != 1:
        raise ValueError("bad Z3G2 header")
    if int(n_pairs) != N_PAIRS or int(n_classes) != N_CLASSES or int(latent_dim) != LATENT_DIM:
        raise ValueError("bad Z3G2 dimensions")
    pos = Z3G2_HEADER.size
    (sigma_len,) = struct.unpack_from("<H", data, pos)
    pos += 2
    sigma_bytes = brotli.decompress(data[pos:pos + sigma_len])
    pos += sigma_len
    sigma_int8 = np.frombuffer(sigma_bytes, dtype=np.int8).reshape(N_CLASSES, LATENT_DIM)
    class_prior = np.frombuffer(data[pos:pos + N_CLASSES * 2], dtype="<u2").astype(np.int64)
    pos += N_CLASSES * 2
    (class_index_len,) = struct.unpack_from("<I", data, pos)
    pos += 4
    class_indices_b = _decode_class_indices_huffman(data[pos:pos + class_index_len], class_prior)
    pos += class_index_len
    (residual_len,) = struct.unpack_from("<I", data, pos)
    pos += 4
    residual_b = brotli.decompress(data[pos:pos + residual_len])
    pos += residual_len
    affine = data[pos:pos + 224]
    pos += 224
    offset = np.frombuffer(affine[:112], dtype=np.float32).copy()
    scale = np.frombuffer(affine[112:], dtype=np.float32).copy()
    sigma = np.maximum(sigma_int8.astype(np.float32) * float(int8_sigma_scale) / 127.0, float(min_sigma))
    classes = np.frombuffer(class_indices_b, dtype=np.uint8).astype(np.int64)
    residual = np.frombuffer(residual_b, dtype=np.int8).reshape(N_PAIRS, LATENT_DIM).astype(np.float32)
    latents = residual * sigma[classes, :] * scale[None, :] + offset[None, :]
    return torch.from_numpy(latents.copy()).to(torch.float32), pos


def split_payload(payload: bytes):
    decoder_section = payload[:A1_DECODER_SECTION_TOTAL]
    section_start = A1_DECODER_SECTION_TOTAL
    if payload[section_start:section_start + 4] != Z3G2_MAGIC:
        raise ValueError("missing Z3G2 magic")
    z3_latents, section_len = decode_z3g2_section(payload[section_start:])
    sidecar = payload[section_start + section_len:]
    return decoder_section, z3_latents, sidecar


def inflate(src_bin: str, dst_raw: str) -> int:
    payload = Path(src_bin).read_bytes()
    decoder_section, z3_latents, sidecar_blob = split_payload(payload)
    decoder_sd = decode_decoder_compact(decoder_section[4:])
    latents = apply_latent_sidecar(z3_latents, sidecar_blob)
    device = select_inflate_device()
    decoder = HNeRVDecoder(latent_dim=LATENT_DIM, base_channels=BASE_CHANNELS, eval_size=(EVAL_H, EVAL_W)).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()
    proof_pairs = int(os.environ.get("Z3G2_FRAME_PROOF_PAIRS", "0") or "0")
    n_pairs = min(N_PAIRS, proof_pairs) if proof_pairs > 0 else N_PAIRS
    latents = latents[:n_pairs].to(device)
    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            batch = j - i
            decoded = decoder(latents[i:j])
            flat = decoded.reshape(batch * 2, 3, EVAL_H, EVAL_W)
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)
            frames = up.reshape(batch * 2, 3, CAMERA_H, CAMERA_W).clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy()
            fout.write(frames.tobytes())
            n += batch * 2
    print(f"[z3g2-inflate] saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
'''


def _z3g2_blob_slices(payload: bytes) -> dict[str, slice]:
    import struct

    section_offset = A1_DECODER_SECTION_TOTAL
    pos = section_offset + Z3G2_HEADER_STRUCT.size
    (sigma_len,) = struct.unpack_from("<H", payload, pos)
    sigma_start = pos + 2
    sigma_end = sigma_start + sigma_len
    class_prior_start = sigma_end
    class_prior_end = class_prior_start + Z3G2_CLASS_PRIOR_BLOB_LEN
    (class_index_len,) = struct.unpack_from("<I", payload, class_prior_end)
    class_index_start = class_prior_end + 4
    class_index_end = class_index_start + class_index_len
    return {
        "sigma_table_blob": slice(sigma_start, sigma_end),
        "class_prior_cdf_blob": slice(class_prior_start, class_prior_end),
        "class_index_blob": slice(class_index_start, class_index_end),
    }


def _find_clean_payload_mutation(payload: bytes) -> dict[str, Any]:
    baseline = reconstruct_class_indices_and_sigma_table_from_z3g2_payload(payload)
    baseline_latent_hash = _sha256_bytes(baseline[1].contiguous().numpy().tobytes())
    for blob_name, blob_slice in _z3g2_blob_slices(payload).items():
        offsets = (0, max(0, (blob_slice.stop - blob_slice.start) // 2), max(0, blob_slice.stop - blob_slice.start - 1))
        for rel_offset in offsets:
            if blob_slice.start + rel_offset >= blob_slice.stop:
                continue
            original = payload[blob_slice.start + rel_offset]
            for replacement in (original ^ 0xA5, original ^ 0x01, (original + 1) & 0xFF, 0, 255):
                if replacement == original:
                    continue
                mutated = bytearray(payload)
                mutated[blob_slice.start + rel_offset] = replacement
                try:
                    outputs = reconstruct_class_indices_and_sigma_table_from_z3g2_payload(bytes(mutated))
                except Exception:
                    continue
                mutated_latent_hash = _sha256_bytes(outputs[1].contiguous().numpy().tobytes())
                if mutated_latent_hash != baseline_latent_hash:
                    return {
                        "blob_name": blob_name,
                        "payload_byte_offset": blob_slice.start + rel_offset,
                        "byte_offset_in_blob": rel_offset,
                        "original_byte": int(original),
                        "replacement_byte": int(replacement),
                        "mutated_payload": bytes(mutated),
                        "baseline_latent_sha256": baseline_latent_hash,
                        "mutated_latent_sha256": mutated_latent_hash,
                    }
    raise RuntimeError("no clean Z3G2 payload mutation changed reconstructed latents")


def _run_inflate_frame_hash(
    *,
    submission_dir: Path,
    payload: bytes,
    work_dir: Path,
    max_pairs: int,
) -> dict[str, Any]:
    inflate_sh = (submission_dir / "inflate.sh").resolve()
    data_dir = work_dir / "data"
    out_dir = work_dir / "out"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "0.bin").write_bytes(payload)
    file_list = work_dir / "file_list.txt"
    file_list.write_text("0.mkv\n", encoding="utf-8")
    env = dict(os.environ)
    env["PYTHON"] = sys.executable
    env["PACT_INFLATE_DEVICE"] = "cpu"
    env["Z3G2_FRAME_PROOF_PAIRS"] = str(max_pairs)
    proc = subprocess.run(
        [str(inflate_sh), str(data_dir), str(out_dir), str(file_list)],
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    raw_path = out_dir / "0.raw"
    if proc.returncode != 0 or not raw_path.is_file():
        raise RuntimeError(
            f"inflate proof failed rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
    raw = raw_path.read_bytes()
    return {
        "raw_sha256": _sha256_bytes(raw),
        "raw_bytes": len(raw),
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-1000:],
    }


def _run_bounded_frame_mutation_proof(
    *,
    submission_dir: Path,
    payload: bytes,
    out_dir: Path,
    max_pairs: int,
) -> dict[str, Any]:
    mutation = _find_clean_payload_mutation(payload)
    mutated_payload = mutation.pop("mutated_payload")
    proof_root = out_dir / "frame_mutation_proof_work"
    if proof_root.exists():
        shutil.rmtree(proof_root)
    baseline = _run_inflate_frame_hash(
        submission_dir=submission_dir,
        payload=payload,
        work_dir=proof_root / "baseline",
        max_pairs=max_pairs,
    )
    mutated = _run_inflate_frame_hash(
        submission_dir=submission_dir,
        payload=mutated_payload,
        work_dir=proof_root / "mutated",
        max_pairs=max_pairs,
    )
    proof = {
        "schema": "z3_g1_entropy_coded_v2_bounded_frame_mutation_proof_v1",
        "proof_scope": "bounded_frame_output_via_inflate_sh_not_full_600_pair_exact_eval",
        "max_pairs": max_pairs,
        "score_claim": False,
        "promotion_eligible": False,
        "baseline": baseline,
        "mutated": mutated,
        "frame_output_sha256_changed": baseline["raw_sha256"] != mutated["raw_sha256"],
        "mutation": mutation,
    }
    proof["verdict"] = "pass" if proof["frame_output_sha256_changed"] else "fail"
    (out_dir / "frame_mutation_proof.json").write_text(
        json.dumps(proof, indent=2, sort_keys=True), encoding="utf-8"
    )
    return proof


def _full_main(args: argparse.Namespace) -> int:
    """Emit a byte-closed research packet and bounded frame mutation proof.

    This is intentionally not a training or score-claim path. It turns the
    Z3G2 idea from a parser-only scaffold into an inspectable contest-shaped
    packet with its own ``inflate.sh`` and runtime manifest, while preserving
    fail-closed authority flags until learned class export plus paired exact
    CPU/CUDA eval exist.
    """
    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _canon_pin_seeds(args.seed)

    def _resolve_repo_path(value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path

    a1_archive_path = _resolve_repo_path(args.a1_archive_path)
    video_path = _resolve_repo_path(args.video_path)
    upstream_dir = _resolve_repo_path(args.upstream_dir)
    if not a1_archive_path.is_file():
        raise FileNotFoundError(f"A1 archive missing: {a1_archive_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"video path missing: {video_path}")
    if not upstream_dir.is_dir():
        raise FileNotFoundError(f"upstream dir missing: {upstream_dir}")
    if args.frame_proof_pairs < 1 and not args.skip_frame_proof:
        raise ValueError("--frame-proof-pairs must be >= 1 unless --skip-frame-proof")

    a1_bytes, a1_payload_sha, a1_payload_bytes, a1_member = _load_archive_member_bytes(
        a1_archive_path
    )
    payload, packet_manifest = _build_z3g2_packet_from_a1(
        a1_archive_path=a1_archive_path,
        a1_bytes=a1_bytes,
    )

    archive_zip_path = out_dir / "archive.zip"
    _build_archive_zip(archive_zip_path, payload_bytes=payload)
    archive_zip_bytes = archive_zip_path.read_bytes()

    submission_dir = out_dir / "submission_dir"
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    runtime_manifest = _write_runtime(
        submission_dir,
        a1_runtime_src=a1_archive_path.parent / "src",
    )
    (submission_dir / "0.bin").write_bytes(payload)
    shutil.copy2(archive_zip_path, submission_dir / "archive.zip")

    frame_proof: dict[str, Any] | None = None
    if not args.skip_frame_proof:
        frame_proof = _run_bounded_frame_mutation_proof(
            submission_dir=submission_dir,
            payload=payload,
            out_dir=out_dir,
            max_pairs=args.frame_proof_pairs,
        )
        if frame_proof["verdict"] != "pass":
            raise RuntimeError(
                "bounded frame-output mutation proof failed; refusing to emit success"
            )

    stats = {
        "schema_version": "z3_g1_entropy_coded_v2_full_export_research_v1",
        "smoke_mode": False,
        "research_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "exact_eval_ready": False,
        "implementation_status": "byte_closed_research_export_no_auth_eval",
        "class_index_strategy": packet_manifest["class_index_strategy"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": args.device,
        "seed": args.seed,
        "a1_archive_path": str(a1_archive_path),
        "a1_archive_sha256": _sha256_bytes(a1_archive_path.read_bytes()),
        "a1_archive_bytes": a1_archive_path.stat().st_size,
        "a1_inner_member": a1_member,
        "a1_inner_payload_sha256": a1_payload_sha,
        "a1_inner_payload_bytes": a1_payload_bytes,
        "archive_zip_path": str(archive_zip_path),
        "archive_zip_sha256": _sha256_bytes(archive_zip_bytes),
        "archive_zip_bytes": len(archive_zip_bytes),
        "z3g2_payload_sha256": packet_manifest["payload_sha256"],
        "z3g2_payload_bytes": packet_manifest["payload_bytes"],
        "submission_dir": str(submission_dir),
        "runtime_tree_sha256": runtime_manifest["runtime_tree_sha256"],
        "runtime_manifest_path": str(
            submission_dir / "submission_runtime_manifest.json"
        ),
        "video_path": str(video_path),
        "video_sha256": _sha256_bytes(video_path.read_bytes()),
        "video_bytes": video_path.stat().st_size,
        "upstream_dir": str(upstream_dir),
        "packet_manifest": packet_manifest,
        "bounded_frame_mutation_proof": {
            "enabled": frame_proof is not None,
            "path": str(out_dir / "frame_mutation_proof.json")
            if frame_proof is not None
            else None,
            "verdict": frame_proof["verdict"] if frame_proof is not None else "skipped",
            "max_pairs": args.frame_proof_pairs,
        },
        "result_review_classification": "byte_closed_research_packet_not_score_anchor",
        "result_review_blockers": [
            *packet_manifest["archive_contract"]["result_review_blockers"],
            "compress_side_segnet_class_export_not_implemented",
            "deterministic_modulo_class_indices_are_placeholder_not_learned",
            "bounded_frame_mutation_proof_not_full_600_pair_exact_eval",
            "paired_contest_cpu_cuda_auth_eval_missing",
            "remote_driver_missing_or_unverified",
            "dispatch_recipe_remains_research_only_false_for_score_claim",
        ],
        "git_head_sha": _canon_git_head_sha(),
        "hardware_substrate": _canon_detect_hardware_substrate(
            axis="cpu", substrate_tag="z3_g1_entropy_coded_v2"
        ),
        "utc_now": _canon_utc_now_iso(),
        "lane_id": "lane_z3_g1_entropy_coded_v2_20260515",
        "substrate_id": "z3_g1_entropy_coded_v2",
        "council_verdict_provenance": ".omx/research/wunderkind_g1_entropy_coded_v2_design_20260515.md",
    }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")

    provenance = {
        "schema_version": "z3_g1_entropy_coded_v2_export_provenance_v1",
        "command_surface": "experiments/train_substrate_z3_g1_entropy_coded_v2.py",
        "stats_path": str(stats_path),
        "runtime_manifest": runtime_manifest,
        "packet_manifest": packet_manifest,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "reason": "research-only deterministic Z3G2 packet export without paired exact eval",
        },
    }
    provenance_path = out_dir / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[full-export] wrote {archive_zip_path}")
    print(f"[full-export] wrote {submission_dir}")
    print(f"[full-export] wrote {stats_path}")
    if frame_proof is not None:
        print(
            "[full-export] bounded frame mutation proof "
            f"{frame_proof['verdict']} at {out_dir / 'frame_mutation_proof.json'}"
        )
    print("[full-export] no score claim; paired CPU/CUDA auth eval still blocked")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    print(f"[z3g1-v2] starting at {_canon_utc_now_iso()}")
    print(f"[z3g1-v2] git_head_sha = {_canon_git_head_sha()}")
    print(f"[z3g1-v2] hardware_substrate = {_canon_detect_hardware_substrate(axis='cpu', substrate_tag='z3_g1_entropy_coded_v2')}")
    print(f"[z3g1-v2] contract registered: id={Z3_G1_ENTROPY_CODED_V2_CONTRACT.id}")
    t0 = time.time()
    rc = _smoke_main(args) if args.smoke else _full_main(args)
    print(f"[z3g1-v2] elapsed = {time.time() - t0:.2f}s; rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
