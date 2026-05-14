#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pre-dispatch frame-parity + no-op pixel-delta proofs for HDM8 postfilter modes.

Per codex frame-exploit research (.omx/research/segnet_posenet_frame_exploit_latest_research_20260514_codex.md
section 7 "Adversarial Failure Modes") any HDM8 selector packet MUST emit these
two proofs BEFORE GPU dispatch:

1. **Frame-parity proof** — first-frame-only modes (``even_*``) must leave every
   ODD-indexed (frame_1 of each pair, i.e. the frame SegNet sees) inflated frame
   byte-identical to the ``none`` baseline. This verifies the
   ``submissions/hdm8_film_grain_sidecar/inflate.py:_apply_postfilter`` even/odd
   gating is structurally correct against the SegNet last-frame contract
   (``upstream/modules.py:103-109`` ``x[:, -1, ...]``).

2. **No-op pixel-delta proof** — every non-``none`` mode must produce a NONZERO
   aggregate pixel delta vs ``none`` AFTER uint8 clamp/round. A mode that
   passes the parser/validator but emits zero pixel-delta (e.g. amplitude
   underflows uint8) is a no-op and would burn GPU spend for zero score
   movement. Sister of Catalog #105 (``check_gate7_no_op_provenance``).

This is a CPU-or-MPS LOCAL proxy probe; tagged ``axis="local-<device>-proof"``,
``score_claim=false``, ``promotion_eligible=false``. The proof is a runtime
contract proof, NOT a score claim. Per CLAUDE.md "Apples-to-apples evidence
discipline" the proof axis stays advisory until the same contract is verified
on contest-CUDA.

Usage::

    .venv/bin/python tools/probe_hdm8_postfilter_frame_parity_and_no_op.py \
        --archive experiments/results/pr106_r2_hdm7_hlm2_hdm8_candidate_20260514_codex/exact_eval_static_release_surface/archive.zip \
        --runtime-template submissions/hdm8_film_grain_sidecar \
        --modes none,even_grain_chroma:1.0,even_grain_chroma:2.0,even_rgb_bias:2,-1,-1 \
        --n-pairs 32 \
        --device cpu \
        --output-json experiments/results/hdm8_film_grain_selector_dispatch_20260514/frame_parity_no_op_proof.json

The proof JSON is consumed by ``tools/build_hdm8_film_grain_sidecar_packet.py``
(or any downstream packet builder) before declaring a packet ready for exact
CUDA dispatch.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr106_r2_hdm7_hlm2_hdm8_candidate_20260514_codex/"
    "exact_eval_static_release_surface/archive.zip"
)
DEFAULT_RUNTIME_TEMPLATE = REPO_ROOT / "submissions/hdm8_film_grain_sidecar"
PROOF_SCHEMA = "hdm8_postfilter_frame_parity_and_no_op_proof_v1"
CAMERA_H, CAMERA_W = 874, 1164


def _load_runtime_modules(runtime_dir: Path) -> Any:
    src_dir = runtime_dir / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"runtime src/ missing: {src_dir}")
    # Insert runtime src/ at the FRONT of sys.path so we pick up the
    # submission's pinned codec/model/pr101_grammar modules.
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    # Force-evict any cached siblings so each invocation gets the right runtime.
    for name in ("codec", "model", "pr101_grammar"):
        sys.modules.pop(name, None)
    inflate_path = runtime_dir / "inflate.py"
    spec = importlib.util.spec_from_file_location(
        "hdm8_film_grain_sidecar_inflate", inflate_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {inflate_path}")
    inflate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inflate)
    return inflate


def _read_archive_payload(archive_path: Path) -> bytes:
    with zipfile.ZipFile(archive_path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise ValueError(
                f"expected one archive member in {archive_path}, found {len(infos)}"
            )
        return zf.read(infos[0].filename)


def _decode_pairs_to_rgb_float(
    inflate: Any, archive_bytes: bytes, *, n_pairs: int, device: torch.device
) -> tuple[torch.Tensor, int]:
    """Decode the HNeRV archive to (frames, 3, H, W) float at camera resolution.

    Returns ``(frames_bchw_float, available_pairs)``. The output is BEFORE any
    postfilter, BEFORE clamp/round/uint8 — i.e. the canonical inflate.py
    intermediate tensor at line 663-666.
    """
    format_id, pr106_bytes, sidecar_blob, framing_meta, _embedded_cfg = (
        inflate.parse_sidecar_archive_with_selector(archive_bytes)
    )
    # Import lazily so we use the runtime's pinned codec/model.
    from codec import parse_packed_archive  # type: ignore[import-not-found]
    from model import HNeRVDecoder  # type: ignore[import-not-found]

    decoder_sd, latents, meta = parse_packed_archive(pr106_bytes)
    if format_id == inflate.SIDECAR_FORMAT_BROTLI:
        if sidecar_blob:
            dim_arr, delta_q_arr = inflate.decode_brotli_sidecar(sidecar_blob)
        else:
            dim_arr = np.full(inflate.PR101_SCHEMA.n_pairs, inflate.NO_OP_DIM, dtype=np.uint8)
            delta_q_arr = np.zeros(inflate.PR101_SCHEMA.n_pairs, dtype=np.int8)
    else:
        dim_arr, delta_q_arr = inflate.decode_pr101_grammar_sidecar(
            sidecar_blob, framing_meta
        )
    inflate.apply_sidecar_corrections(latents, dim_arr, delta_q_arr)

    decoder = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()
    latents = latents.to(device)

    available_pairs = int(meta["n_pairs"])
    use_pairs = min(int(n_pairs), available_pairs)
    eval_h, eval_w = meta["eval_size"]
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for i in range(0, use_pairs, 16):
            j = min(i + 16, use_pairs)
            decoded = decoder(latents[i:j])  # (B, 2, 3, eval_h, eval_w)
            flat = decoded.reshape((j - i) * 2, 3, eval_h, eval_w)
            up = F.interpolate(
                flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
            )
            chunks.append(up)
    return torch.cat(chunks, dim=0), available_pairs


def _frames_to_uint8_bytes(frames_bchw_float: torch.Tensor) -> bytes:
    """Apply the canonical clamp(0,255).round().to(uint8) cast and return raw bytes.

    Matches ``submissions/hdm8_film_grain_sidecar/inflate.py:667-670``.
    """
    arr = (
        frames_bchw_float.clamp(0, 255)
        .permute(0, 2, 3, 1)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return arr.tobytes()


def _per_frame_sha256(frames_bytes: bytes, n_frames: int) -> list[str]:
    """Compute one sha256 per frame given concatenated (N, H, W, 3) uint8 bytes."""
    if n_frames == 0:
        return []
    frame_size = len(frames_bytes) // n_frames
    if frame_size * n_frames != len(frames_bytes):
        raise ValueError(
            f"per-frame size mismatch: total_bytes={len(frames_bytes)} "
            f"n_frames={n_frames}; expected exact divisor"
        )
    return [
        hashlib.sha256(frames_bytes[i * frame_size : (i + 1) * frame_size]).hexdigest()
        for i in range(n_frames)
    ]


def _apply_mode_global(
    inflate: Any, frames_bchw_float: torch.Tensor, mode: str
) -> torch.Tensor:
    """Apply the postfilter mode to a flat (N, C, H, W) batch with frame_start=0.

    Matches the canonical inflate.py flow at line 666:
    ``apply_postfilter_config(up, config, pair_start=i)`` with ``i=0`` and the
    full batch as one shot. For selector mode the caller routes through
    ``apply_postfilter_config``; for fixed modes we call ``apply_postfilter``
    directly with ``frame_start=0`` so even/odd parity gating uses real frame
    indices (frame 0, 1, 2, ..., N-1).
    """
    return inflate.apply_postfilter(frames_bchw_float, mode, frame_start=0)


def _verify_frame_parity(
    *,
    baseline_frame_shas: list[str],
    mode_frame_shas: list[str],
    n_frames: int,
    mode: str,
) -> dict[str, Any]:
    """Verify SegNet-side (odd-index) parity for first-frame-only modes.

    The HDM8 pair contract: frame ``2i`` is frame_0 of pair i; frame ``2i+1`` is
    frame_1 of pair i (the frame SegNet sees). ``even_*`` modes touch only frame
    indices where ``idx % 2 == 0`` per ``inflate.py`` line 517. Therefore EVERY
    ODD-indexed frame must hash-match the ``none`` baseline for SegNet to remain
    exactly unchanged.
    """
    is_even_only = mode.startswith("even_") or mode == "none"
    odd_indices = [i for i in range(n_frames) if i % 2 == 1]
    even_indices = [i for i in range(n_frames) if i % 2 == 0]
    odd_mismatches = [
        i for i in odd_indices if mode_frame_shas[i] != baseline_frame_shas[i]
    ]
    even_mismatches = [
        i for i in even_indices if mode_frame_shas[i] != baseline_frame_shas[i]
    ]
    if mode == "none":
        contract = "no_change_anywhere"
        passed = len(odd_mismatches) == 0 and len(even_mismatches) == 0
    elif is_even_only:
        # even_* modes MUST leave odd frames byte-identical (SegNet null) AND
        # SHOULD touch at least one even frame (otherwise it's a no-op masked
        # as a parity-preserving mode).
        contract = "first_frame_only_segnet_null"
        passed = len(odd_mismatches) == 0
    elif mode.startswith("odd_"):
        contract = "last_frame_only_posenet_null_segnet_risk"
        passed = len(even_mismatches) == 0  # even frames must be untouched
    else:
        contract = "both_frames_touched"
        passed = True  # parity contract not applicable
    return {
        "mode": mode,
        "parity_contract": contract,
        "passed": bool(passed),
        "n_odd_frames_checked": len(odd_indices),
        "n_even_frames_checked": len(even_indices),
        "n_odd_mismatches": len(odd_mismatches),
        "n_even_mismatches": len(even_mismatches),
        "first_odd_mismatch_idx": odd_mismatches[0] if odd_mismatches else None,
        "first_even_mismatch_idx": even_mismatches[0] if even_mismatches else None,
    }


def _verify_no_op_pixel_delta(
    *,
    baseline_bytes: bytes,
    mode_bytes: bytes,
    mode: str,
) -> dict[str, Any]:
    """Verify each non-``none`` mode actually changes at least one inflated byte.

    A mode that survives the validator but emits zero pixel-delta (e.g. amplitude
    underflows uint8 round-tripping) is a no-op and would burn GPU spend.
    """
    baseline = np.frombuffer(baseline_bytes, dtype=np.uint8)
    candidate = np.frombuffer(mode_bytes, dtype=np.uint8)
    if baseline.shape != candidate.shape:
        return {
            "mode": mode,
            "passed": False,
            "blocker": "shape_mismatch",
            "baseline_bytes": int(baseline.size),
            "mode_bytes": int(candidate.size),
        }
    diff = baseline.astype(np.int32) - candidate.astype(np.int32)
    n_bytes_changed = int((diff != 0).sum())
    sum_abs_delta = int(np.abs(diff).sum())
    max_abs_delta = int(np.abs(diff).max()) if baseline.size else 0
    is_no_op = (mode == "none")
    if mode == "none":
        passed = (n_bytes_changed == 0)
    else:
        passed = (n_bytes_changed > 0)
    return {
        "mode": mode,
        "passed": bool(passed),
        "is_no_op_expected": bool(is_no_op),
        "n_bytes_changed": n_bytes_changed,
        "fraction_bytes_changed": (
            n_bytes_changed / baseline.size if baseline.size else 0.0
        ),
        "sum_abs_delta": sum_abs_delta,
        "max_abs_delta": max_abs_delta,
    }


def _select_device(name: str) -> torch.device:
    name = name.lower()
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA unavailable")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS unavailable")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported --device: {name!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="HDM8 archive.zip (single-member x payload, format_id=0x02).",
    )
    parser.add_argument(
        "--runtime-template",
        type=Path,
        default=DEFAULT_RUNTIME_TEMPLATE,
        help="Runtime template dir containing inflate.py + src/.",
    )
    parser.add_argument(
        "--modes",
        required=True,
        help="Comma-separated postfilter modes to probe (must include 'none' first).",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=32,
        help="Number of frame pairs to decode (default 32 for ~30s on CPU; "
        "max archive available is 600).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "mps"),
        help="Decode + filter device. MPS is allowed for the PROOF only (it "
        "verifies the runtime contract, not the score).",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="Where to write the proof JSON.",
    )
    return parser.parse_args(argv)


def _parse_modes(spec: str) -> list[str]:
    """Split a comma-separated mode list, honoring rgb_bias triplets.

    ``rgb_bias:<r>,<g>,<b>`` and ``even_rgb_bias:<r>,<g>,<b>`` contain commas
    that are NOT mode-separators. We walk the string character-by-character;
    when we hit ``rgb_bias:`` we know the NEXT TWO commas belong to the
    triplet (so the 3rd comma is the mode separator). Any other comma is a
    mode separator.
    """
    out: list[str] = []
    current = ""
    triplet_commas_left = 0
    i = 0
    while i < len(spec):
        ch = spec[i]
        if ch == ",":
            if triplet_commas_left > 0:
                current += ch
                triplet_commas_left -= 1
                i += 1
                continue
            if current.strip():
                out.append(current.strip())
            current = ""
            i += 1
            continue
        current += ch
        # After each character, check whether the CURRENT token has just
        # opened an ``rgb_bias:`` triplet. We detect by looking at the
        # trailing 9 chars and resetting triplet_commas_left to 2 if so.
        if current.endswith("rgb_bias:") and triplet_commas_left == 0:
            triplet_commas_left = 2
        i += 1
    if current.strip():
        out.append(current.strip())
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    archive_path = args.archive.resolve()
    runtime_template = args.runtime_template.resolve()
    output_json = args.output_json.resolve()
    if not archive_path.exists():
        raise SystemExit(f"archive not found: {archive_path}")
    if not runtime_template.exists():
        raise SystemExit(f"runtime template not found: {runtime_template}")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    modes = _parse_modes(args.modes)
    if not modes:
        raise SystemExit("--modes must list at least one mode")
    if modes[0] != "none":
        raise SystemExit(
            f"--modes must start with 'none' as the baseline; got first={modes[0]!r}"
        )
    # Validate every mode through the runtime's own validator BEFORE we decode.
    inflate = _load_runtime_modules(runtime_template)
    for mode in modes:
        inflate.validate_postfilter_mode(mode)

    device = _select_device(args.device)
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()

    archive_bytes = _read_archive_payload(archive_path)
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    archive_size_bytes = archive_path.stat().st_size

    frames_float, available_pairs = _decode_pairs_to_rgb_float(
        inflate, archive_bytes, n_pairs=args.n_pairs, device=device
    )
    n_frames = frames_float.shape[0]
    if n_frames != 2 * min(args.n_pairs, available_pairs):
        raise SystemExit(
            f"decoded {n_frames} frames; expected 2*{min(args.n_pairs, available_pairs)}"
        )

    # Compute mode bytes + frame SHA256s.
    mode_results: list[dict[str, Any]] = []
    baseline_bytes: bytes | None = None
    baseline_frame_shas: list[str] | None = None
    for mode in modes:
        filtered = _apply_mode_global(inflate, frames_float, mode)
        out_bytes = _frames_to_uint8_bytes(filtered)
        frame_shas = _per_frame_sha256(out_bytes, n_frames)
        if mode == "none":
            baseline_bytes = out_bytes
            baseline_frame_shas = frame_shas
        if baseline_bytes is None or baseline_frame_shas is None:
            raise RuntimeError("internal: baseline must be computed first")
        parity = _verify_frame_parity(
            baseline_frame_shas=baseline_frame_shas,
            mode_frame_shas=frame_shas,
            n_frames=n_frames,
            mode=mode,
        )
        no_op = _verify_no_op_pixel_delta(
            baseline_bytes=baseline_bytes,
            mode_bytes=out_bytes,
            mode=mode,
        )
        mode_results.append(
            {
                "mode": mode,
                "aggregate_sha256": hashlib.sha256(out_bytes).hexdigest(),
                "frame_parity_proof": parity,
                "no_op_pixel_delta_proof": no_op,
                "passed_all_proofs": parity["passed"] and no_op["passed"],
            }
        )

    finished_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    proof = {
        "schema": PROOF_SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "axis": f"local-{device.type}-proof",
        "started_at_utc": started_utc,
        "finished_at_utc": finished_utc,
        "archive_path": str(archive_path.relative_to(REPO_ROOT)),
        "archive_size_bytes": archive_size_bytes,
        "archive_sha256": archive_sha256,
        "runtime_template_path": str(runtime_template.relative_to(REPO_ROOT)),
        "device": device.type,
        "n_pairs_decoded": min(args.n_pairs, available_pairs),
        "n_pairs_archive_total": available_pairs,
        "n_frames": n_frames,
        "modes": mode_results,
        "all_modes_passed": all(item["passed_all_proofs"] for item in mode_results),
        "blocker_modes": [
            item["mode"] for item in mode_results if not item["passed_all_proofs"]
        ],
        "evidence_tag": f"[local-{device.type}-proof; runtime-contract-only; not a score claim]",
        "next_required_actions": [
            "if all_modes_passed=true, packet builder may proceed to build candidate archives",
            "if blocker_modes is non-empty, fix inflate.py parity gating OR remove the offending modes from the palette BEFORE exact CUDA dispatch",
            "exact CUDA score claim still requires experiments/contest_auth_eval.py --device cuda; this proof does NOT replace that",
        ],
    }
    output_json.write_text(json.dumps(proof, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(proof, indent=2, sort_keys=True))
    if not proof["all_modes_passed"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
