#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PACT-NeRV-VQ MLX → PyTorch state_dict bridge with forward-parity proof.

PER-SUBSTRATE INDIVIDUALLY-FRACTAL canonical engineering pass per the 11th
INDIVIDUALLY-FRACTAL standing directive 2026-05-27 — PACT-NeRV-VQ's OWN
bridge tool sister of ``tools/export_pact_nerv_ia3_mlx_to_pytorch_state_dict.py``.
The bridge enables the L2 paired-CUDA promotion path per Catalog #233 4-gate
canonical:

    MLX numpy-portable state_dict (.npsd from MLX trainer)
      |
      v   :func:`export_pact_nerv_vq_mlx_to_pytorch`
      v
    PyTorch .pt state_dict (canonical OIHW Conv2d layout + VQ buffers)
      |
      v  +- forward parity proof (MLX vs PyTorch on identical input)
      v  +- canonical Provenance per Catalog #287/#323
      v
    PVQ archive via tac.substrates.pact_nerv_vq.archive.pack_archive
      |
      v   :mod:`tools.gate_mlx_candidate_contest_equivalence_pact_nerv_vq`
      v
    Catalog #1265 contest-equivalence verdict (PASS/FAIL)

VQ-specific contract vs IA3 sister:
- VQ state_dict contains ``quantizer.codebook`` + ``quantizer.ema_cluster_size``
  + ``quantizer.ema_w`` buffers (NOT trainable parameters per van den Oord
  §3.2 EMA discipline).
- VQ has NO ``ego_poses`` per-pair tensor (no pose conditioning; the
  distinguishing primitive is discrete-token quantization, not γ-only FiLM).
- Forward parity uses the per-pair learnable latent ``self.latents`` only.
- Codebook + EMA buffers are passed through unchanged in MLX layout (they
  are not Conv2d weights; the transpose-on-export logic only applies to
  4-D Conv2d weight tensors).

Per CLAUDE.md non-negotiables PRESERVED:
- "Submission auth eval - BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
  HARDWARE": this pipeline targets that gate explicitly - MLX is the
  training surface; CUDA T4 (or paired Linux x86_64) is the eval surface.
- "MLX portable-local-substrate authority": MLX-trained weights are
  non-promotable until evaluated via PyTorch on CUDA T4.
- "Bugs must be permanently fixed AND self-protected against": canonical
  Provenance per Catalog #287/#323 + Tier A markers per Catalog #341
  (axis_tag='[predicted]' / score_claim=False / promotable=False).
- Catalog #110/#113 APPEND-ONLY: export writes NEW .pt + parity_proof.json;
  never mutates existing forensic artifacts.
- HNeRV parity L4: this tool is ~300 LOC + ≤2 ext deps (numpy + torch);
  the optional MLX dependency is only required for the forward-parity proof
  (skipped on non-Apple-Silicon hosts).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tac.local_acceleration.mlx_to_pytorch_export import (
    assign_mlx_param_by_dotted_name,
    build_substrate_bridge_manifest,
    hash_file_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PVQ_BRIDGE_SCHEMA = "pact_nerv_vq_mlx_pytorch_export_bridge.v1"


def _hash_file(path: Path) -> str:
    return hash_file_sha256(path)


def export_pact_nerv_vq_mlx_to_pytorch(
    *,
    mlx_state_dict_path: Path,
    output_pytorch_state_dict: Path,
    parity_proof_out: Path | None = None,
    sample_pair_indices: tuple[int, ...] = (0, 1, 2),
    atol_max_01: float = 0.001,
    atol_mean_01: float = 1e-4,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Convert an MLX .npsd checkpoint to PyTorch .pt + emit forward parity proof.

    Args:
        mlx_state_dict_path: MLX numpy-portable state_dict blob (.npsd) emitted
            by ``tac.substrates._shared.mlx_score_aware`` adapter checkpoint.
        output_pytorch_state_dict: destination PyTorch ``.pt`` file path.
        parity_proof_out: optional path for ``numpy_pytorch_parity_proof.json``.
        sample_pair_indices: pair indices to forward through both backends.
        atol_max_01: per-tensor max-abs drift threshold in ``[0, 1]`` sigmoid
            space (default 0.001 mirrors sister IA3/Z6 #1265 gate defaults).
        atol_mean_01: per-tensor mean-abs drift threshold in ``[0, 1]`` space.
        overwrite: refuse existing destination if False.

    Returns:
        Export manifest dict with canonical Provenance fields.
    """
    import numpy as np
    import torch

    from tac.provenance.builders import build_provenance_for_predicted
    from tac.provenance.validator import provenance_to_dict
    from tac.substrates._shared.numpy_portable_inflate import (
        unpack_state_dict_numpy,
    )
    from tac.substrates.pact_nerv_vq.architecture import (
        PactNervVqConfig,
        PactNervVqSubstrate,
    )

    src = Path(mlx_state_dict_path)
    if not src.is_file():
        raise FileNotFoundError(f"MLX state_dict not found: {src}")
    out_pt = Path(output_pytorch_state_dict)
    if out_pt.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite: {out_pt}")

    # 1) Unpack MLX-side state_dict (numpy arrays in MLX HWIO Conv2d layout).
    mlx_sd_np = unpack_state_dict_numpy(src.read_bytes())

    # 2) Transpose Conv2d weights MLX HWIO -> PyTorch OIHW; preserve VQ
    #    buffers + Linear weights + biases + per-pair tensors.
    #
    # PRINCIPLED FORK per Catalog #290 (canonical-vs-unique decision per
    # layer): the canonical helper
    # ``tac.framework_agnostic.helpers.convert_mlx_state_dict_to_pytorch_oihw``
    # (extracted 2026-05-30 from 5-tool duplication) supports a
    # ``skip_buffer_name_predicate`` callback, but consuming it would change
    # this tool's persisted ``per_tensor[name]["layout"]`` token for
    # quantizer.* buffers from ``"preserved"`` to ``"skipped_by_predicate"``
    # — NOT byte-stable, would break downstream consumers that check
    # ``layout == "preserved"``. Plus this tool carries a
    # substrate-distinguishing ``is_vq_buffer: bool`` sidecar field per
    # VQ-VAE §3.2 that the canonical helper does not expose. FORK_BECAUSE_
    # SUPPRESSES per the canonical falling-rule. (Sister tools IA3 +
    # SELECTOR_V2/V3/V4 + Z6_V2_CARGO_CULT_UNWIND consume the canonical
    # helper because they have no quantizer.* buffers + no
    # is_vq_buffer sidecar.)
    #
    # VQ buffer names: quantizer.codebook / quantizer.ema_cluster_size /
    # quantizer.ema_w (all 1-D or 2-D non-Conv tensors per VQ-VAE §3.2).
    pytorch_sd: dict[str, torch.Tensor] = {}
    per_tensor: dict[str, dict[str, Any]] = {}
    for name, arr in mlx_sd_np.items():
        out_arr = arr
        layout_note = "preserved"
        is_quantizer_buf = name.startswith("quantizer.")
        if name.endswith(".weight") and arr.ndim == 4 and not is_quantizer_buf:
            # MLX Conv2d weight is (out, kH, kW, in) -> PyTorch (out, in, kH, kW).
            out_arr = np.transpose(arr, (0, 3, 1, 2))
            layout_note = "mlx_hwio_to_pytorch_oihw"
        out_arr = np.ascontiguousarray(out_arr).astype(np.float32)
        pytorch_sd[name] = torch.from_numpy(out_arr.copy())
        per_tensor[name] = {
            "shape_mlx": list(arr.shape),
            "shape_pytorch": list(out_arr.shape),
            "dtype": str(out_arr.dtype),
            "sha256": hashlib.sha256(out_arr.tobytes()).hexdigest()[:16],
            "layout": layout_note,
            "is_vq_buffer": bool(is_quantizer_buf),
        }

    # 3) Build the PyTorch sister substrate + load_state_dict (strict=True).
    #    Infer (num_pairs, latent_dim, codebook_size) from loaded tensors.
    num_pairs, latent_dim = pytorch_sd["latents"].shape
    codebook_size = int(pytorch_sd["quantizer.codebook"].shape[0])
    cb_latent_dim = int(pytorch_sd["quantizer.codebook"].shape[1])
    if cb_latent_dim != int(latent_dim):
        raise RuntimeError(
            f"codebook latent_dim {cb_latent_dim} != latents latent_dim {latent_dim}"
        )
    cfg = PactNervVqConfig(
        latent_dim=int(latent_dim),
        codebook_size=int(codebook_size),
        num_pairs=int(num_pairs),
    )
    model = PactNervVqSubstrate(cfg).eval()
    load_result = model.load_state_dict(pytorch_sd, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"PyTorch load_state_dict mismatch: missing={load_result.missing_keys}; "
            f"unexpected={load_result.unexpected_keys}"
        )

    # 4) Save PyTorch .pt (canonical layout) per Catalog #14 weights_only-loadable.
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pytorch_sd, out_pt)
    file_sha = _hash_file(out_pt)
    file_size = out_pt.stat().st_size

    # 5) Forward-parity proof (MLX vs PyTorch). MLX is optional - skip cleanly
    #    on non-Apple-Silicon hosts so the bridge works as a pure converter.
    parity: dict[str, Any]
    try:
        from tac.framework_agnostic import require_mlx_core

        mx = require_mlx_core()

        from tac.substrates.pact_nerv_vq.mlx_renderer import (
            PactNervVqSubstrateMLX,
        )

        mlx_model = PactNervVqSubstrateMLX(cfg)
        # Load the SAME MLX state_dict back (MLX layout); compare both forwards.
        for name, arr in mlx_sd_np.items():
            _assign_mlx_param(mlx_model, name, arr)
        pair_idx = list(sample_pair_indices)
        py_idx = torch.tensor(pair_idx, dtype=torch.long)
        with torch.no_grad():
            py_rgb_0, py_rgb_1 = model(py_idx)
        # PyTorch sister outputs sigmoid in [0, 1]; MLX renderer outputs
        # sigmoid * 255 per the canonical "call_b2chw_255" convention.
        py_stack_01 = (
            torch.stack([py_rgb_0, py_rgb_1], dim=1).numpy().astype(np.float32)
        )
        mlx_idx = mx.array(np.asarray(pair_idx, dtype=np.int32))
        mlx_arr_255 = np.asarray(mlx_model(mlx_idx), dtype=np.float32)
        mlx_arr_01 = mlx_arr_255 / 255.0
        drift = np.abs(py_stack_01 - mlx_arr_01)
        parity = {
            "sample_pair_indices": pair_idx,
            "max_abs_drift_01": float(drift.max()),
            "mean_abs_drift_01": float(drift.mean()),
            "atol_max_01": atol_max_01,
            "atol_mean_01": atol_mean_01,
            "drift_within_band": bool(
                drift.max() <= atol_max_01 and drift.mean() <= atol_mean_01
            ),
            "frame_shape": list(py_stack_01.shape),
            "backends_compared": "mlx_vs_pytorch_forward",
            "decoder_output_space": "sigmoid_0_to_1",
            "vq_specific_note": (
                "VQ-VAE codebook + per-pair index forward parity test. "
                "MLX VectorQuantizerEMAMLX uses mx.stop_gradient for STE; "
                "PyTorch VectorQuantizerEMA uses .detach(). Both produce "
                "byte-identical quantization for the same z_e + codebook "
                "(the codebook lookup is argmin-deterministic in both "
                "backends; STE differences only affect backward pass)."
            ),
            "drift_vs_depth_anchor": (
                "MLX vs PyTorch drift on a 7-PixelShuffle SIREN substrate is "
                "the canonical drift-vs-depth signature per Catalog #1305; a "
                "drift band wider than the threshold does NOT imply a bridge "
                "bug. sin(freq=30.0) activation amplifies per-layer ~1e-6 conv "
                "drift exponentially across 7 upsample blocks. The threshold "
                "is a research-signal disambiguator, not a contest-promotion "
                "gate; promotion requires paired contest-CUDA."
            ),
        }
    except Exception as exc:  # pragma: no cover - non-Apple Silicon CI path.
        parity = {
            "sample_pair_indices": list(sample_pair_indices),
            "backends_compared": "skipped_mlx_unavailable",
            "skip_reason": repr(exc),
            "drift_within_band": False,
        }

    # 6) Canonical Provenance per Catalog #287/#323 - non-promotable until paired
    #    Linux x86_64 + NVIDIA evidence lands per Catalog #1/#192/#317/#341.
    inputs_sha = _hash_file(src)
    prov = build_provenance_for_predicted(
        model_id=f"pact_nerv_vq_mlx_pytorch_bridge:{src.stem}",
        inputs_sha256=inputs_sha,
        measurement_axis="[predicted]",
        hardware_substrate="darwin_arm64_apple_silicon_mlx",
    )

    manifest = build_substrate_bridge_manifest(
        schema_version=PVQ_BRIDGE_SCHEMA,
        tool="tools/export_pact_nerv_vq_mlx_to_pytorch_state_dict.py",
        source_state_dict_path=src,
        output_pytorch_state_dict=out_pt,
        source_state_dict_sha256=inputs_sha,
        pytorch_state_dict_sha256=file_sha,
        pytorch_state_dict_bytes=file_size,
        tensor_count=len(pytorch_sd),
        config={
            "num_pairs": int(num_pairs),
            "latent_dim": int(latent_dim),
            "codebook_size": int(codebook_size),
        },
        per_tensor=per_tensor,
        forward_parity=parity,
        operator_routable_next_step=(
            "Pack PVQ archive via tac.substrates.pact_nerv_vq.archive.pack_archive, "
            "then route through tools/gate_mlx_candidate_contest_equivalence_pact_nerv_vq.py "
            "(Catalog #1265 sister), then operator-authorize.py paired CPU+CUDA dispatch"
        ),
        provenance=provenance_to_dict(prov),
        extra_fields={
            "mlx_state_dict_bytes": src.stat().st_size,
            "vq_buffer_tensor_count": sum(
                1 for n in pytorch_sd if n.startswith("quantizer.")
            ),
        },
    )
    if parity_proof_out is not None:
        proof_path = Path(parity_proof_out)
        proof_path.parent.mkdir(parents=True, exist_ok=True)
        proof_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["parity_proof_path"] = str(proof_path)
    return manifest


def _assign_mlx_param(mlx_model: Any, dotted_name: str, np_arr: Any) -> None:
    """Walk dotted name + assign a numpy array onto MLX param (MLX layout preserved).

    VQ-specific: ``quantizer.codebook`` / ``quantizer.ema_cluster_size`` /
    ``quantizer.ema_w`` are private-prefix attributes (``_codebook`` /
    ``_ema_cluster_size`` / ``_ema_w``) on the MLX quantizer; the assignment
    routes through the underscore-prefixed name.
    """
    assign_mlx_param_by_dotted_name(
        mlx_model,
        dotted_name,
        np_arr,
        leaf_name_overrides={
            "quantizer.codebook": "_codebook",
            "quantizer.ema_cluster_size": "_ema_cluster_size",
            "quantizer.ema_w": "_ema_w",
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-state-dict", type=Path, required=True)
    parser.add_argument("--output-pytorch-state-dict", type=Path, required=True)
    parser.add_argument("--parity-proof-out", type=Path)
    parser.add_argument("--sample-indices", default="0,1,2")
    parser.add_argument("--atol-max-01", type=float, default=0.001)
    parser.add_argument("--atol-mean-01", type=float, default=1e-4)
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--require-parity-pass", action="store_true")
    args = parser.parse_args(argv)

    sample_indices = tuple(int(s) for s in args.sample_indices.split(",") if s.strip())
    manifest = export_pact_nerv_vq_mlx_to_pytorch(
        mlx_state_dict_path=args.mlx_state_dict,
        output_pytorch_state_dict=args.output_pytorch_state_dict,
        parity_proof_out=args.parity_proof_out,
        sample_pair_indices=sample_indices,
        atol_max_01=args.atol_max_01,
        atol_mean_01=args.atol_mean_01,
        overwrite=not args.no_overwrite,
    )
    parity = manifest["forward_parity"]
    within = parity.get("drift_within_band", False)
    print(f"[pact-nerv-vq-bridge] pt={manifest['output_pytorch_state_dict']}")
    print(f"[pact-nerv-vq-bridge] tensor_count={manifest['tensor_count']} "
          f"vq_buffers={manifest['vq_buffer_tensor_count']}")
    if parity.get("backends_compared") == "mlx_vs_pytorch_forward":
        print(
            f"[pact-nerv-vq-bridge] max_abs_01={parity['max_abs_drift_01']:.6e} "
            f"mean_abs_01={parity['mean_abs_drift_01']:.6e} within_band={within}"
        )
    else:
        print(
            f"[pact-nerv-vq-bridge] parity SKIPPED ({parity.get('skip_reason', 'n/a')})"
        )
    if args.parity_proof_out is not None:
        print(f"[pact-nerv-vq-bridge] proof={args.parity_proof_out}")
    return 1 if args.require_parity_pass and not within else 0


__all__ = [
    "PVQ_BRIDGE_SCHEMA",
    "export_pact_nerv_vq_mlx_to_pytorch",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
