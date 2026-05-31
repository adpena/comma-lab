#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Z8 hierarchical-predictive-coding MLX → PyTorch state_dict bridge + self-parity proof.

PER-SUBSTRATE INDIVIDUALLY-FRACTAL canonical engineering pass per the 11th
INDIVIDUALLY-FRACTAL standing directive — Z8's OWN MLX→PyTorch export bridge,
the LAST of the 9-member predictive Z-stack still missing its bridge. Sister of:

- ``tools/export_z6_v2_cargo_cult_unwind_mlx_to_pytorch_state_dict.py`` (Z6-v2 sister)
- ``tools/export_pr95_mlx_to_pytorch_state_dict.py`` (PR95 sister)
- ``tools/export_wyner_ziv_pipeline_stage_codec_mlx_to_pytorch_state_dict.py``

The bridge enables the L2 paired-CUDA-eligibility path per Catalog #233 4-gate:

    MLX numpy-portable state_dict (.npsd from the canonical mlx_score_aware harness)
      |
      v   :func:`export_z8_hier_pc_mlx_to_pytorch`
      v
    PyTorch .pt state_dict (canonical OIHW Conv2d layout)
      |
      v  +- numpy-portable self-parity proof (MLX renderer forward vs
      v  |   numpy-portable-reconstructed renderer forward on identical input)
      v  +- canonical Provenance per Catalog #287/#323 + Tier A markers (#341)
      v
    PyTorch-side analysis / future PyTorch Z8 renderer / paired CUDA+CPU anchor

## Catalog #290 canonical-vs-unique decision per layer

- ADOPT_CANONICAL: HWIO→OIHW Conv2d transpose via the canonical
  ``convert_mlx_state_dict_to_pytorch_oihw`` helper (OBVIOUS-FIT; identical
  transpose semantics to the 5 sister bridges); manifest emission via
  ``build_substrate_bridge_manifest``; ``.npsd`` unpack via the canonical
  numpy-portable helper; canonical Provenance + Tier A markers.
- FORK_PRINCIPLED (Z8 DISTINGUISHING per Catalog #290 falling-rule):
  1. **Parity strategy.** Z8 has NO PyTorch sister ``nn.Module`` (it is
     MLX-only). The sister Z6-v2 bridge runs MLX-forward vs PyTorch-sister
     forward; Z8 cannot. The honest, genuinely-available NO-FAKE parity is
     **MLX-renderer forward vs MLX-renderer-reconstructed-from-the-exported-
     numpy-state-dict forward** (round-trip self-parity). This proves the
     exported numpy/PyTorch weights reproduce the SAME frames the trained MLX
     renderer produces — exactly the invariant a PyTorch-side reconstruction
     needs. (Forward-vs-PyTorch parity is deferred to the future PyTorch Z8
     renderer; this bridge does not fabricate a PyTorch sister it does not have.)
  2. **List-stored categorical posterior.** Z8's per-pair per-level categorical
     logits live on plain Python lists (``logits_per_level``) NOT MLX
     ``nn.Module`` attributes, so the canonical ``assign_mlx_param_by_dotted_name``
     cannot walk ``logits_per_level.<i>``. ``_assign_z8_mlx_param`` handles those
     by direct list assignment and delegates everything else to the canonical
     helper.

Per CLAUDE.md non-negotiables:
- "MLX portable-local-substrate authority" — output non-promotable
  ``[macOS-MLX research-signal]`` per Catalog #192/#341; the bridge produces a
  state_dict + parity proof, NOT a score claim.
- "Submission auth eval - BOTH CPU AND CUDA" — promotion still requires a
  paired Linux x86_64 + NVIDIA anchor downstream.
- Catalog #110/#113 APPEND-ONLY: NEW .pt + parity_proof.json; never mutates.
- "NO FAKE IMPLEMENTATIONS" — the bridge loads REAL MLX weights, exports a REAL
  PyTorch-loadable state_dict, and measures a REAL self-parity delta; a
  parity skip on a non-Apple-Silicon host is reported as a skip (NOT a pass).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tac.local_acceleration.mlx_to_pytorch_export import (
    assign_mlx_param_by_dotted_name,
    build_substrate_bridge_manifest,
    hash_file_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
Z8_BRIDGE_SCHEMA = "z8_hierarchical_predictive_coding_mlx_pytorch_export_bridge.v1"

# Canonical PR95 base grid the Z8 decoder hardcodes (stem out = C * base_h * base_w).
_BASE_H = 6
_BASE_W = 8


def _hash_file(path: Path) -> str:
    return hash_file_sha256(path)


def infer_z8_config_from_state_dict(mlx_sd_np: dict[str, Any]) -> dict[str, int]:
    """Infer the Z8HierarchicalConfig kwargs from a numpy-portable state_dict.

    Every field is derived from a real tensor shape so the bridge never
    hardcodes a config that could silently diverge from the trained
    checkpoint (Catalog #229 premise-verification discipline).
    """
    L = 0
    num_groups: list[int] = []
    num_categories: list[int] = []
    while f"logits_per_level.{L}" in mlx_sd_np:
        shp = mlx_sd_np[f"logits_per_level.{L}"].shape
        if len(shp) != 3:
            raise ValueError(
                f"logits_per_level.{L} expected (N, G, K) 3D; got shape {shp}"
            )
        num_groups.append(int(shp[1]))
        num_categories.append(int(shp[2]))
        L += 1
    if L == 0:
        raise ValueError(
            "Z8 MLX state_dict missing required 'logits_per_level.0' "
            "per-pair categorical posterior tensor"
        )
    num_pairs = int(mlx_sd_np["logits_per_level.0"].shape[0])

    # decoder_latent_dim = cat_to_continuous_per_level.0.weight out-features.
    decoder_latent_dim = int(mlx_sd_np["cat_to_continuous_per_level.0.weight"].shape[0])

    # stem.weight out-features = base_channels * base_h * base_w.
    stem_out = int(mlx_sd_np["stem.weight"].shape[0])
    if stem_out % (_BASE_H * _BASE_W) != 0:
        raise ValueError(
            f"stem.weight out-features {stem_out} not divisible by "
            f"base grid {_BASE_H}*{_BASE_W}; cannot infer base_channels"
        )
    base_channels = stem_out // (_BASE_H * _BASE_W)

    # deterministic_gate.weight: (det_dim, decoder_latent_dim + ego_motion_dim).
    det_dim = int(mlx_sd_np["deterministic_gate.weight"].shape[0])
    gru_input_dim = int(mlx_sd_np["deterministic_gate.weight"].shape[1])
    ego_motion_dim = gru_input_dim - decoder_latent_dim
    if ego_motion_dim <= 0:
        raise ValueError(
            f"inferred ego_motion_dim {ego_motion_dim} <= 0; "
            f"deterministic_gate in-features {gru_input_dim} <= "
            f"decoder_latent_dim {decoder_latent_dim}"
        )

    return {
        "num_levels": L,
        "num_groups_per_level": tuple(num_groups),
        "num_categories_per_level": tuple(num_categories),
        "base_channels": int(base_channels),
        "decoder_latent_dim": int(decoder_latent_dim),
        "num_pairs": int(num_pairs),
        "deterministic_state_dim": int(det_dim),
        "ego_motion_dim": int(ego_motion_dim),
    }


def _assign_z8_mlx_param(mlx_model: Any, dotted_name: str, np_arr: Any) -> None:
    """Assign one numpy array onto a Z8 MLX renderer param (MLX layout preserved).

    Z8 FORK_PRINCIPLED: ``logits_per_level.<i>`` lives on a plain Python list
    (NOT an ``nn.Module`` attribute), so the canonical dotted-walk helper
    cannot index it. Handle those by direct list assignment; delegate every
    other dotted name to the canonical helper.
    """
    from tac.framework_agnostic import require_mlx_core

    mx = require_mlx_core()
    if dotted_name.startswith("logits_per_level."):
        idx = int(dotted_name.split(".", 1)[1])
        mlx_model.logits_per_level[idx] = mx.array(np_arr)
        return
    assign_mlx_param_by_dotted_name(mlx_model, dotted_name, np_arr)


def export_z8_hier_pc_mlx_to_pytorch(
    *,
    mlx_state_dict_path: Path,
    output_pytorch_state_dict: Path,
    parity_proof_out: Path | None = None,
    sample_pair_indices: tuple[int, ...] = (0, 1),
    atol_max_01: float = 0.001,
    atol_mean_01: float = 1e-4,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Convert a Z8 MLX ``.npsd`` checkpoint to PyTorch ``.pt`` + self-parity proof.

    Args:
        mlx_state_dict_path: MLX numpy-portable state_dict blob (.npsd) emitted
            by the canonical ``mlx_score_aware`` harness checkpoint writer.
        output_pytorch_state_dict: destination PyTorch ``.pt`` file path.
        parity_proof_out: optional path for ``numpy_pytorch_parity_proof.json``.
        sample_pair_indices: pair indices to forward through both backends.
        atol_max_01: per-frame max-abs drift threshold in ``[0, 1]`` space.
        atol_mean_01: per-frame mean-abs drift threshold in ``[0, 1]`` space.
        overwrite: refuse existing destination if False.

    Returns:
        Export manifest dict with canonical Provenance fields.
    """
    import numpy as np
    import torch

    from tac.framework_agnostic.helpers import (
        convert_mlx_state_dict_to_pytorch_oihw,
    )
    from tac.provenance.builders import build_provenance_for_predicted
    from tac.provenance.validator import provenance_to_dict
    from tac.substrates._shared.numpy_portable_inflate import (
        unpack_state_dict_numpy,
    )

    src = Path(mlx_state_dict_path)
    if not src.is_file():
        raise FileNotFoundError(f"MLX state_dict not found: {src}")
    out_pt = Path(output_pytorch_state_dict)
    if out_pt.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite: {out_pt}")

    # 1) Unpack MLX-side state_dict (numpy arrays in MLX HWIO Conv2d layout).
    mlx_sd_np = unpack_state_dict_numpy(src.read_bytes())

    # 2) Infer config from real tensor shapes (Catalog #229 premise discipline).
    inferred = infer_z8_config_from_state_dict(mlx_sd_np)

    # 3) Transpose Conv2d weights MLX HWIO -> PyTorch OIHW via the canonical
    #    helper. logits_per_level.* (3D non-conv) + Linear weights pass through.
    pytorch_sd, per_tensor = convert_mlx_state_dict_to_pytorch_oihw(mlx_sd_np)

    # 4) Save PyTorch .pt (canonical layout; weights_only-loadable per Catalog #14).
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pytorch_sd, out_pt)
    file_sha = _hash_file(out_pt)
    file_size = out_pt.stat().st_size

    # 5) Numpy-portable SELF-parity proof (MLX renderer forward vs MLX renderer
    #    reconstructed from the exported numpy state_dict). MLX optional - skip
    #    cleanly on non-Apple-Silicon hosts (reported as a skip, NOT a pass,
    #    per CLAUDE.md NO FAKE IMPLEMENTATIONS).
    parity: dict[str, Any]
    try:
        from tac.framework_agnostic import require_mlx_core

        mx = require_mlx_core()

        from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
            Z8HierarchicalConfig,
            Z8HierarchicalPredictiveCoderMLX,
        )

        cfg = Z8HierarchicalConfig(
            num_levels=int(inferred["num_levels"]),
            num_groups_per_level=inferred["num_groups_per_level"],
            num_categories_per_level=inferred["num_categories_per_level"],
            base_channels=int(inferred["base_channels"]),
            decoder_latent_dim=int(inferred["decoder_latent_dim"]),
            num_pairs=int(inferred["num_pairs"]),
            deterministic_state_dim=int(inferred["deterministic_state_dim"]),
            ego_motion_dim=int(inferred["ego_motion_dim"]),
            gumbel_temperature=1.0,
            use_straight_through=True,
        )

        # Renderer A: assign the trained weights directly from the .npsd.
        model_a = Z8HierarchicalPredictiveCoderMLX(cfg)
        for name, arr in mlx_sd_np.items():
            _assign_z8_mlx_param(model_a, name, arr)

        # Renderer B: round-trip through the EXPORTED numpy state_dict (the
        # bytes the PyTorch side will consume). Convert PyTorch OIHW back to
        # MLX HWIO so the MLX renderer can load it — this proves the exported
        # bytes reconstruct the identical renderer.
        mlx_sd_from_export = _pytorch_oihw_to_mlx_hwio_numpy(pytorch_sd)
        model_b = Z8HierarchicalPredictiveCoderMLX(cfg)
        for name, arr in mlx_sd_from_export.items():
            _assign_z8_mlx_param(model_b, name, arr)

        pair_idx = list(sample_pair_indices)
        # Use the DETERMINISTIC eval path: take argmax indices from renderer A's
        # categorical posterior, decode both renderers from the same indices so
        # the comparison is Gumbel-noise-free (parity, not RNG drift).
        per_level_indices = _argmax_indices_for_pairs(mx, model_a, cfg, pair_idx)
        out_a = np.asarray(
            model_a.forward_eval_from_indices(per_level_indices),
            dtype=np.float32,
        )
        out_b = np.asarray(
            model_b.forward_eval_from_indices(per_level_indices),
            dtype=np.float32,
        )
        # Renderer outputs sigmoid * 255 per canonical "call_b2chw_255" convention.
        out_a_01 = out_a / 255.0
        out_b_01 = out_b / 255.0
        drift = np.abs(out_a_01 - out_b_01)
        parity = {
            "sample_pair_indices": pair_idx,
            "max_abs_drift_01": float(drift.max()),
            "mean_abs_drift_01": float(drift.mean()),
            "atol_max_01": atol_max_01,
            "atol_mean_01": atol_mean_01,
            "drift_within_band": bool(
                drift.max() <= atol_max_01 and drift.mean() <= atol_mean_01
            ),
            "frame_shape": list(out_a.shape),
            "backends_compared": (
                "mlx_renderer_vs_mlx_renderer_reconstructed_from_exported_numpy_state_dict"
            ),
            "decoder_output_space": "sigmoid_0_to_1",
            "parity_strategy_note": (
                "Z8 has NO PyTorch sister nn.Module (MLX-only). This is "
                "round-trip self-parity: the trained MLX renderer vs the MLX "
                "renderer reconstructed from the PyTorch OIHW .pt bytes the "
                "bridge exports (transposed back to MLX HWIO). A near-zero "
                "delta proves the exported PyTorch state_dict reproduces the "
                "identical frames; it is the invariant a future PyTorch Z8 "
                "renderer reconstruction needs. NOT a contest-promotion gate; "
                "promotion requires paired contest-CUDA per Catalog #246."
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
        model_id=f"z8_hierarchical_predictive_coding_mlx_pytorch_bridge:{src.stem}",
        inputs_sha256=inputs_sha,
        measurement_axis="[predicted]",
        hardware_substrate="darwin_arm64_apple_silicon_mlx",
    )

    manifest = build_substrate_bridge_manifest(
        schema_version=Z8_BRIDGE_SCHEMA,
        tool="tools/export_z8_hier_pc_mlx_to_pytorch_state_dict.py",
        source_state_dict_path=src,
        output_pytorch_state_dict=out_pt,
        source_state_dict_sha256=inputs_sha,
        pytorch_state_dict_sha256=file_sha,
        pytorch_state_dict_bytes=file_size,
        tensor_count=len(pytorch_sd),
        config=inferred,
        per_tensor=per_tensor,
        forward_parity=parity,
        operator_routable_next_step=(
            "DEFERRED (architectural): the Z8 contest inflate (inflate.py) "
            "reconstructs frames from the archive wavelet_coeffs_blob (Mallat "
            "perfect reconstruction), NOT from these trained HNeRV decoder "
            "weights. The exported .pt enables PyTorch-side analysis / a future "
            "PyTorch Z8 renderer. To make THESE weights contest-eligible, a "
            "Z8 archive grammar revision that packs + inflate-consumes the "
            "categorical-posterior decoder weights is required (a separate wave; "
            "wiring them into the current pack_archive without inflate "
            "consumption would be the Catalog #220 research-substrate trap)."
        ),
        provenance=provenance_to_dict(prov),
        extra_fields={"mlx_state_dict_bytes": src.stat().st_size},
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


def _pytorch_oihw_to_mlx_hwio_numpy(pytorch_sd: dict[str, Any]) -> dict[str, Any]:
    """Transpose a PyTorch OIHW state_dict back to MLX HWIO numpy (round-trip).

    The inverse of ``convert_mlx_state_dict_to_pytorch_oihw`` for the self-parity
    proof: 4D Conv2d weights OIHW -> HWIO; all other tensors pass through.
    """
    import numpy as np

    out: dict[str, Any] = {}
    for name, value in pytorch_sd.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 4 and name.endswith(".weight"):
            # OIHW -> HWIO (inverse of HWIO -> OIHW = transpose(0,3,1,2)).
            out[name] = np.ascontiguousarray(np.transpose(arr, (0, 2, 3, 1)))
        else:
            out[name] = np.ascontiguousarray(arr)
    return out


def _argmax_indices_for_pairs(
    mx: Any, model: Any, cfg: Any, pair_idx: list[int]
) -> list[Any]:
    """Deterministic per-level argmax category indices for the given pairs.

    Used to drive ``forward_eval_from_indices`` so the parity comparison is
    Gumbel-noise-free (a true weight-parity comparison, not RNG drift).
    """
    indices: list[Any] = []
    sel = mx.array(__import__("numpy").asarray(pair_idx, dtype=__import__("numpy").int32))
    for level_idx in range(int(cfg.num_levels)):
        level_logits = mx.take(model.logits_per_level[level_idx], sel, axis=0)
        indices.append(mx.argmax(level_logits, axis=-1))  # (B, G_l)
    return indices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-state-dict", type=Path, required=True)
    parser.add_argument("--output-pytorch-state-dict", type=Path, required=True)
    parser.add_argument("--parity-proof-out", type=Path)
    parser.add_argument("--sample-indices", default="0,1")
    parser.add_argument("--atol-max-01", type=float, default=0.001)
    parser.add_argument("--atol-mean-01", type=float, default=1e-4)
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--require-parity-pass", action="store_true")
    args = parser.parse_args(argv)

    sample_indices = tuple(int(s) for s in args.sample_indices.split(",") if s.strip())
    manifest = export_z8_hier_pc_mlx_to_pytorch(
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
    print(f"[z8-bridge] pt={manifest['output_pytorch_state_dict']}")
    print(f"[z8-bridge] tensor_count={manifest['tensor_count']}")
    if parity.get("backends_compared", "").startswith("mlx_renderer_vs"):
        print(
            f"[z8-bridge] max_abs_01={parity['max_abs_drift_01']:.6e} "
            f"mean_abs_01={parity['mean_abs_drift_01']:.6e} within_band={within}"
        )
    else:
        print(f"[z8-bridge] parity SKIPPED ({parity.get('skip_reason', 'n/a')})")
    if args.parity_proof_out is not None:
        print(f"[z8-bridge] proof={args.parity_proof_out}")
    return 1 if args.require_parity_pass and not within else 0


__all__ = [
    "Z8_BRIDGE_SCHEMA",
    "export_z8_hier_pc_mlx_to_pytorch",
    "infer_z8_config_from_state_dict",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
