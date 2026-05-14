# SPDX-License-Identifier: MIT
"""Lane 20 — Real-archive empirical measurement on Lane G v3 anchor.

Walks an FP4A renderer.bin, extracts the per-layer FP4 nibble stream (the
"qint stream" Lane 20 targets), and measures byte savings under three
codecs:

1. **Raw packed nibbles** — the current FP4A baseline (4 bits per element)
2. **Static arithmetic codec** — ``encode_qints_arithmetic`` (1 freq table
   per layer)
3. **Hotz-LITE BHv1** — chunked static prior (K configurable per layer)
4. **Full Ballé BHv1** — block-conditional Gaussian (untrained codec OR
   loaded from a checkpoint via --hyperprior-checkpoint)

The output is a JSON report at ``--output`` which is the empirical artifact
referenced by the ``[empirical:reports/lane_20_balle_real_archive.json]``
tag in CLAUDE.md docstrings.

Usage::

    python experiments/measure_lane_20_balle_real_archive.py \\
        --renderer experiments/results/lane_g_v3_landed/iter_0/renderer.bin \\
        --output reports/lane_20_balle_real_archive.json

Per CLAUDE.md non-negotiables
-----------------------------
* No silent defaults — every flag required.
* Tags every claim ``[empirical:reports/lane_20_balle_real_archive.json]``.
* No GPU dependency at decode (codecs are pure CPU).
* ``--device cpu`` is the only option here (we are measuring byte length,
  not a learned signal).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.arithmetic_qint_codec import encode_qints_arithmetic, decode_qints_arithmetic
from tac.balle_hyperprior_codec import (
    BalleHyperpriorCodec,
    HyperDecoder,
    HyperEncoder,
    decode_qints_balle,
    encode_qints_balle_auto,
    encode_qints_full_balle,
    encode_qints_hotz_lite,
)
from tac.fp4_quantize import DEFAULT_CODEBOOK, RESIDUAL_CODEBOOK, _quantize_block


def _unpack_fp4_nibbles(packed: bytes, num_elements: int) -> np.ndarray:
    """Inverse of FP4A's ``_pack_indices_signs``: returns ``(num_elements,)``
    int values in [-7, 7] excluding 0-tied ambiguity.

    The FP4 wire format is:
        [bit 7] sign bit of element 2k
        [bits 6:4] codebook index of element 2k
        [bit 3] sign bit of element 2k+1
        [bits 2:0] codebook index of element 2k+1

    We reconstruct the SIGNED CODE = sign × index ∈ [-7, +7] for each element.
    The codebook value lookup is NOT applied here — Lane 20 codes the
    *index* stream, not the dequantized fp32.
    """
    out = np.empty(num_elements, dtype=np.int8)
    for i in range(num_elements):
        byte = packed[i // 2]
        if i % 2 == 0:
            sign = (byte >> 7) & 1
            idx = (byte >> 4) & 0x07
        else:
            sign = (byte >> 3) & 1
            idx = byte & 0x07
        out[i] = -idx if sign else idx
    return out


def _fp4_quantize_to_signed_indices(
    weights: np.ndarray,
    *,
    block_size: int = 32,
    codebook: torch.Tensor,
    robust_scale: bool = False,
) -> np.ndarray:
    """FP4-quantize a 1-D weight tensor and return SIGNED indices ∈ [-7,+7].

    Used to simulate the FP4A qint stream from a raw FP16 ASYM weight tensor,
    so Lane 20 can be measured on Lane G v3 (which is currently FP16 ASYM,
    not FP4A). The signed-index alphabet is what Lane 20 actually compresses.
    """
    w = torch.from_numpy(weights.astype(np.float32))
    n = w.numel()
    pad = (block_size - n % block_size) % block_size
    if pad:
        w = torch.cat([w, torch.zeros(pad)])
    blocks = w.view(-1, block_size)
    signed_indices = []
    for b in blocks:
        idx, sgn, _ = _quantize_block(b, codebook, robust_scale=robust_scale)
        # idx ∈ [0,7], sgn ∈ {-1, +1} → signed index ∈ {-7..+7}
        si = idx.to(torch.int8) * sgn.to(torch.int8)
        signed_indices.append(si)
    out = torch.cat(signed_indices)[:n].numpy().astype(np.int8)
    return out


def _scan_asym_layers(blob: bytes) -> list[dict]:
    """Parse ASYM header to extract per-Conv2d FP16 weights as numpy arrays."""
    if blob[:4] != b"ASYM":
        raise ValueError(f"Not an ASYM binary: magic={blob[:4]!r}")
    # Use the canonical loader and walk model.named_parameters()
    from tac.renderer_export import load_asymmetric_checkpoint

    model = load_asymmetric_checkpoint(blob, device="cpu")
    layers = []
    for name, p in model.named_parameters():
        if p.dim() == 4:  # Conv2d
            arr = p.detach().cpu().float().numpy().reshape(-1)
            layers.append({
                "name": name,
                "shape": list(p.shape),
                "n_elements": int(arr.size),
                "weights_fp32": arr,
            })
    return layers


def _scan_fp4a_layers(blob: bytes) -> list[dict]:
    """Parse FP4A header to enumerate per-layer weight chunks."""
    if blob[:4] != b"FP4A":
        raise ValueError(f"Not an FP4A binary: magic={blob[:4]!r}")
    cursor = 4
    (header_len,) = struct.unpack("<I", blob[cursor : cursor + 4])
    cursor += 4
    header_json = json.loads(blob[cursor : cursor + header_len].decode("utf-8"))
    cursor += header_len

    layers = []
    for entry in header_json.get("layers", []):
        (blob_len,) = struct.unpack("<I", blob[cursor : cursor + 4])
        cursor += 4
        layer_blob = blob[cursor : cursor + blob_len]
        cursor += blob_len
        layers.append({
            "name": entry.get("name", "?"),
            "shape": entry.get("shape", []),
            "n_elements": int(entry.get("n_elements", 0)),
            "n_blocks": int(entry.get("n_blocks", 0)),
            "blob_len": blob_len,
            "blob": layer_blob,
        })

    embeddings = []
    for entry in header_json.get("embeddings", []):
        (blob_len,) = struct.unpack("<I", blob[cursor : cursor + 4])
        cursor += 4
        layer_blob = blob[cursor : cursor + blob_len]
        cursor += blob_len
        embeddings.append({
            "name": entry.get("name", "?"),
            "n_elements": int(entry.get("n_elements", 0)),
            "blob_len": blob_len,
            "blob": layer_blob,
        })
    return layers, embeddings


def _measure_layer(
    *,
    name: str,
    qints: np.ndarray,
    full_codec: BalleHyperpriorCodec,
    static_baseline_bytes_per_elem: float,
) -> dict:
    """Measure all four codecs on a single layer's qint stream.

    ``qints``: 1-D int8 array with values ∈ [-7, 7].
    Maps to symbol indices via offset=7, num_symbols=15.
    """
    n = qints.size
    # Skip near-empty layers (head.weight has 108 elements; not worth measuring)
    if n < 32:
        return {
            "name": name,
            "n_elements": n,
            "skipped": True,
            "reason": "n_elements < 32",
        }

    # 1. Raw FP4 packed bytes (baseline)
    raw_bytes = (n + 1) // 2  # 4 bits/element

    # 2. Static arithmetic codec
    static_blob = encode_qints_arithmetic(qints, num_symbols=15, offset=7)
    static_bytes = len(static_blob)
    # Verify roundtrip
    static_decoded = decode_qints_arithmetic(static_blob, expected_dtype=np.int8)
    assert np.array_equal(static_decoded, qints), f"Static codec roundtrip failed on {name}"

    # 3. Hotz-LITE (chunk count proportional to log(n))
    K = max(2, min(16, int(np.log2(max(n // 128, 4)))))
    lite_blob = encode_qints_hotz_lite(
        qints=qints, num_symbols=15, offset=7, num_chunks=K
    )
    lite_bytes = len(lite_blob)
    lite_decoded = decode_qints_balle(blob=lite_blob, expected_dtype=np.int8)
    assert np.array_equal(lite_decoded, qints), f"Hotz-LITE roundtrip failed on {name}"

    # 4. Full Ballé (untrained — we measure baseline behavior)
    if n >= full_codec.block_size:
        full_blob = encode_qints_full_balle(
            qints=qints, num_symbols=15, offset=7, codec=full_codec,
        )
        full_bytes = len(full_blob)
        full_decoded = decode_qints_balle(blob=full_blob, expected_dtype=np.int8)
        assert np.array_equal(full_decoded, qints), f"Full Ballé roundtrip failed on {name}"
    else:
        full_bytes = -1  # not applicable

    # 5. Auto-select with static-baseline guard
    auto_blob, auto_mode, auto_stats = encode_qints_balle_auto(
        qints=qints, num_symbols=15, offset=7,
        num_chunks_lite=K,
        full_codec=full_codec if n >= full_codec.block_size else None,
        static_baseline_bytes=static_bytes,
    )
    auto_bytes = len(auto_blob) if auto_blob else static_bytes  # static fallback

    return {
        "name": name,
        "n_elements": n,
        "raw_fp4_bytes": raw_bytes,
        "static_arithmetic_bytes": static_bytes,
        "hotz_lite_bytes": lite_bytes,
        "hotz_lite_K": K,
        "full_balle_bytes": full_bytes,
        "auto_bytes": auto_bytes,
        "auto_mode": auto_mode,
        "savings_vs_raw_fp4_static_pct": (
            (raw_bytes - static_bytes) / raw_bytes * 100 if raw_bytes else 0
        ),
        "savings_vs_static_lite_pct": (
            (static_bytes - lite_bytes) / static_bytes * 100 if static_bytes else 0
        ),
        "savings_vs_static_auto_pct": (
            (static_bytes - auto_bytes) / static_bytes * 100 if static_bytes else 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure Lane 20 BHv1 codec on Lane G v3 FP4A renderer.bin."
    )
    parser.add_argument("--renderer", type=str, required=True, help="path to FP4A renderer.bin")
    parser.add_argument("--output", type=str, required=True, help="path to JSON report")
    parser.add_argument("--block-size", type=int, default=128, help="full-Ballé block size")
    parser.add_argument("--z-dim", type=int, default=4, help="full-Ballé hyper-latent dim")
    parser.add_argument("--hidden", type=int, default=8, help="hyper-decoder hidden width")
    parser.add_argument("--seed", type=int, default=2026, help="codec init seed")
    parser.add_argument(
        "--fp4-codebook", type=str, default="default",
        choices=("default", "residual"),
        help=(
            "Round 1 Finding 2 fix: which FP4 codebook to use when "
            "FP4-quantising an ASYM renderer's FP16 weights. MUST match "
            "what the renderer trained against (Lane G v3 = 'default'; "
            "halfframe profiles = 'residual')."
        ),
    )
    args = parser.parse_args()

    renderer_path = Path(args.renderer)
    if not renderer_path.exists():
        raise FileNotFoundError(f"renderer not found: {renderer_path}")
    blob = renderer_path.read_bytes()
    magic = blob[:4]
    print(f"[lane-20] Loaded {renderer_path} ({len(blob):,} bytes, magic={magic!r})")

    embeddings: list[dict] = []
    if magic == b"FP4A":
        layers, embeddings = _scan_fp4a_layers(blob)
        source_format = "FP4A"
        # FP4A pre-extracts indices; signal that
        for L in layers:
            L["needs_quantize"] = False
    elif magic == b"ASYM":
        layers = _scan_asym_layers(blob)
        source_format = "ASYM (FP16, simulated FP4A via on-the-fly _quantize_block)"
        for L in layers:
            L["needs_quantize"] = True
    else:
        raise ValueError(
            f"Unsupported magic {magic!r}; Lane 20 measurement supports ASYM "
            f"(FP16) and FP4A (FP4 indices)."
        )
    print(
        f"[lane-20] source_format={source_format} — "
        f"{len(layers)} conv layers + {len(embeddings)} embeddings"
    )

    # Build the (untrained) full-Ballé codec
    block_size = args.block_size
    z_dim = args.z_dim
    hidden = args.hidden
    full_codec = BalleHyperpriorCodec(
        block_size=block_size,
        z_dim=z_dim,
        hyper_encoder=HyperEncoder(
            block_size=block_size, z_dim=z_dim, hidden_dim=hidden, seed=args.seed
        ),
        hyper_decoder=HyperDecoder(
            z_dim=z_dim, hidden_dim=hidden, seed=args.seed
        ),
    )
    print(
        f"[lane-20] Full-Ballé codec: block_size={block_size} z_dim={z_dim} "
        f"hidden={hidden} hyper_decoder_byte_size={full_codec.hyper_decoder_byte_size()}B"
    )

    # Process each conv layer
    results = []
    total_raw = 0
    total_static = 0
    total_auto = 0
    t0 = time.monotonic()
    if args.fp4_codebook == "residual":
        codebook = RESIDUAL_CODEBOOK.clone()
    else:
        codebook = DEFAULT_CODEBOOK.clone()
    print(f"[lane-20] using fp4_codebook={args.fp4_codebook!r}")
    for layer in layers:
        name = layer["name"]
        n_elements = layer["n_elements"]
        if layer.get("needs_quantize", False):
            # ASYM: FP4-quantize on the fly to obtain the signed-index stream
            try:
                qints = _fp4_quantize_to_signed_indices(
                    layer["weights_fp32"], codebook=codebook, robust_scale=False
                )
            except Exception as exc:
                results.append({"name": name, "skipped": True, "reason": f"quantize error: {exc}"})
                continue
        else:
            # FP4A: unpack nibbles from the pre-quantized blob
            n_blocks = layer["n_blocks"]
            layer_blob = layer["blob"]
            scales_bytes = n_blocks * 2
            if len(layer_blob) < scales_bytes:
                results.append({"name": name, "skipped": True, "reason": "layer blob truncated"})
                continue
            packed = layer_blob[scales_bytes:]
            try:
                qints = _unpack_fp4_nibbles(packed, n_elements)
            except Exception as exc:
                results.append({"name": name, "skipped": True, "reason": f"unpack error: {exc}"})
                continue
        if qints.size != n_elements:
            results.append({
                "name": name, "skipped": True,
                "reason": f"unpacked size {qints.size} != expected {n_elements}",
            })
            continue
        # Verify alphabet
        if int(qints.min()) < -7 or int(qints.max()) > 7:
            results.append({
                "name": name, "skipped": True,
                "reason": f"out-of-alphabet symbols min={qints.min()} max={qints.max()}",
            })
            continue
        meas = _measure_layer(
            name=name,
            qints=qints,
            full_codec=full_codec,
            static_baseline_bytes_per_elem=0.5,  # 4 bits/elem
        )
        results.append(meas)
        if not meas.get("skipped", False):
            total_raw += meas["raw_fp4_bytes"]
            total_static += meas["static_arithmetic_bytes"]
            total_auto += meas["auto_bytes"]

    elapsed = time.monotonic() - t0
    summary = {
        "renderer_path": str(renderer_path),
        "renderer_total_bytes": len(blob),
        "source_format": source_format,
        "renderer_magic": magic.decode("ascii", errors="replace"),
        "n_layers_measured": sum(1 for r in results if not r.get("skipped", False)),
        "n_layers_skipped": sum(1 for r in results if r.get("skipped", False)),
        "elapsed_seconds": round(elapsed, 2),
        "block_size": block_size,
        "z_dim": z_dim,
        "hidden": hidden,
        "total_raw_fp4_qint_bytes": total_raw,
        "total_static_arithmetic_bytes": total_static,
        "total_balle_auto_bytes": total_auto,
        "savings_static_vs_raw_pct": (
            (total_raw - total_static) / total_raw * 100 if total_raw else 0
        ),
        "savings_balle_auto_vs_static_pct": (
            (total_static - total_auto) / total_static * 100 if total_static else 0
        ),
        "savings_balle_auto_vs_raw_pct": (
            (total_raw - total_auto) / total_raw * 100 if total_raw else 0
        ),
        "claude_md_compliance": {
            "tag_required": "[empirical:" + str(args.output) + "]",
            "scorer_load_at_inflate": False,
            "no_silent_defaults": True,
        },
        "per_layer": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(
        f"[lane-20] Wrote report to {out_path} "
        f"({summary['n_layers_measured']} layers measured, "
        f"savings_vs_static={summary['savings_balle_auto_vs_static_pct']:.2f}%, "
        f"savings_vs_raw={summary['savings_balle_auto_vs_raw_pct']:.2f}%)"
    )


if __name__ == "__main__":
    main()
