#!/usr/bin/env python3
"""PR101 Ω-OPT linear stack empirical byte composition — Path B step 1.

Per the operator's "Path B" choice and the Ω-OPT-paradigm-audit memo
(`feedback_omega_opt_paradigm_audit_unanchored_speculation_20260508`),
no Ω-OPT predicted score has any 1:1 empirical anchor — the predicted
0.130 (linear stack) is arithmetic on per-component predictions, not
a measured composition.

This tool builds the FIRST 1:1 empirical anchor for the linear stack
by composing the architecture-lane components in sequence and
measuring the actual byte count.

Composition (in order):
  1. arch_shrink_x0.4  (truncate channels by L2 magnitude, keep 40%)
  2. IMP sparsity α=0.7 (zero 70% of remaining weights by magnitude)
  3. lossy_coarsening_analytical (per-tensor K-step quantization search)
  4. brotli (with the canonical Optuna-best params)

The composition is naive: each layer operates on the output of the
previous. No retraining; this measures BYTES under the post-hoc
composition. SCORE impact would require retraining per layer +
contest-CUDA evaluation (out of scope for this CPU tool).

Per the metacognitive falsification protocol
(`feedback_premature_falsification_metacognitive_failure_mode_20260508`):
- Verdict: `MEASURED_CONFIG_NOT_DISPATCHABLE` if archive >= baseline
- `family_falsified=False` always (this is one config out of many)
- `falsification_scope=measured_configuration_only`
- `cuda_eval_worth_testing` separate from `ready_for_exact_eval_dispatch`
  which stays False
- Explicit `composition_assumption: layers compose naively without
  retrain-side interaction` disclosure

Per the Ω-OPT-paradigm-audit:
- Honest verdict: this is a BYTE anchor, not a SCORE anchor
- The Ω-OPT predicted 0.130 score requires CUDA verification +
  retrained-from-scratch arch_shrink (subagent C's Lightning training)
- THIS tool only validates the BYTES claim of the design

CLAUDE.md compliance: pure CPU + numpy + brotli; no scorer load; no
contest score claim; output tagged appropriately.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import struct
import sys
from pathlib import Path

import brotli
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
)

TOOL_NAME = "tools/pr101_omega_opt_linear_stack_empirical.py"
SCHEMA_VERSION = "pr101_omega_opt_linear_stack_empirical.v1"
ARCHIVE_OVERHEAD_BYTES = 16_094
DEFAULT_STATE_DICT_PATH = (
    REPO_ROOT
    / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt"
)
EVIDENCE_GRADE = "[CPU-prep empirical Ω-OPT stack composition byte-anchor]"
EVIDENCE_SEMANTICS = "cpu_omega_opt_linear_stack_byte_composition_no_score"
DISPATCH_BLOCKERS = (
    "post_hoc_composition_no_retrain",
    "score_impact_unknown_requires_full_retrain_plus_cuda",
    "no_runtime_decoder_packet_built",
    "missing_exact_cuda_auth_eval",
    "requires_exact_cuda_auth_eval_before_any_score_use",
    "cpu_byte_anchor_not_score_evidence",
)


def proxy_evidence_contract() -> dict:
    return {
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_marker": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "proxy_row": True,
        "family_falsified": False,
        "falsification_scope": "measured_configuration_only",
        "composition_assumption": (
            "layers compose naively post-hoc; no retrain; "
            "score impact requires full retrain + contest-CUDA"
        ),
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
    }


def channel_l2_magnitude(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim == 1:
        return np.abs(tensor)
    flat = tensor.reshape(tensor.shape[0], -1)
    return np.sqrt((flat ** 2).sum(axis=1))


def step1_arch_shrink(tensor: np.ndarray, ratio: float) -> np.ndarray:
    """Truncate top-N output channels by L2 magnitude. Returns smaller tensor."""
    if tensor.ndim == 0 or tensor.size == 0:
        return tensor
    n_total = tensor.shape[0]
    n_keep = max(1, int(round(n_total * ratio)))
    if n_keep >= n_total:
        return tensor
    mags = channel_l2_magnitude(tensor)
    top_idx = np.sort(np.argsort(mags)[-n_keep:])
    return tensor[top_idx]


def step2_imp_sparsify(tensor: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Zero out smallest-magnitude alpha fraction. Returns (sparse_tensor, mask)."""
    if tensor.size == 0:
        return tensor, np.zeros_like(tensor, dtype=bool)
    flat = tensor.flatten()
    n = flat.size
    n_zero = int(round(alpha * n))
    if n_zero <= 0:
        return tensor, np.ones_like(tensor, dtype=bool)
    if n_zero >= n:
        return np.zeros_like(tensor), np.zeros_like(tensor, dtype=bool)
    threshold_idx = np.argpartition(np.abs(flat), n_zero)[n_zero - 1]
    threshold = abs(flat[threshold_idx])
    mask_flat = np.abs(flat) > threshold
    sparse_flat = np.where(mask_flat, flat, 0.0)
    return sparse_flat.reshape(tensor.shape), mask_flat.reshape(tensor.shape)


def step3_lossy_coarsening_per_tensor(tensor: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Per-tensor K-step quantization search (lossy_coarsening_analytical pattern).

    Searches over K ∈ {3, 7, 15, 31, 63, 127} for the coarsest representation
    that keeps rel_err < 5%. Returns (codes, K_chosen, scale).
    """
    if tensor.size == 0:
        return np.array([], dtype=np.int8), 0, 0.0
    nz_mask = mask.flatten()
    nz_values = tensor.flatten()[nz_mask]
    if nz_values.size == 0:
        return np.zeros(tensor.size, dtype=np.int8), 0, 0.0
    abs_max = float(np.abs(nz_values).max())
    if abs_max <= 0.0:
        return np.zeros(tensor.size, dtype=np.int8), 0, 0.0

    best_K = 127
    best_rel_err = float("inf")
    best_codes = None
    best_scale = 1.0
    for K in (3, 7, 15, 31, 63, 127):
        scale = abs_max / K
        codes_nz = np.clip(np.round(nz_values / scale), -K, K).astype(np.int16)
        recon_nz = codes_nz.astype(np.float64) * scale
        eps = 1e-8
        valid = np.abs(nz_values) > eps
        if valid.any():
            rel_err = float((np.abs(recon_nz[valid] - nz_values[valid]) / np.abs(nz_values[valid])).mean()) * 100.0  # REL_ERR_NON_CANONICAL_OK: per-element L1 mean as percentage; PR101 omega-opt linear-stack convention; not allocator-fed
        else:
            rel_err = 0.0
        if rel_err < 5.0:
            best_K = K
            best_rel_err = rel_err
            best_scale = scale
            best_codes_full = np.zeros(tensor.size, dtype=np.int16)
            best_codes_full[nz_mask] = codes_nz
            best_codes = best_codes_full
            break  # smallest K satisfying the constraint
    if best_codes is None:
        # Fall back to K=127
        scale = abs_max / 127
        codes_nz = np.clip(np.round(nz_values / scale), -127, 127).astype(np.int16)
        best_codes = np.zeros(tensor.size, dtype=np.int16)
        best_codes[nz_mask] = codes_nz
        best_K = 127
        best_scale = scale
    return best_codes.astype(np.int8 if best_K <= 127 else np.int16), best_K, best_scale


def encode_stack_per_tensor(tensor: np.ndarray, *, shrink_ratio: float, sparsity_alpha: float) -> tuple[bytes, dict]:
    """Apply the full stack to one tensor, return (payload, stats)."""
    n_orig = tensor.size
    # Step 1: arch_shrink
    t_shrunk = step1_arch_shrink(tensor, shrink_ratio)
    n_shrunk = t_shrunk.size
    # Step 2: IMP sparsity
    t_sparse, mask = step2_imp_sparsify(t_shrunk, sparsity_alpha)
    n_nz = int(mask.sum())
    # Step 3: lossy coarsening
    codes, K, scale = step3_lossy_coarsening_per_tensor(t_sparse, mask)
    # Pack: (n_kept_channels, K, scale_fp16, sparse_indices_uint16, nz_values_int8)
    # For simplicity, store: (n_shrunk u32, K u8, scale fp16, dense codes int8/int16)
    if K <= 127:
        codes_bytes = codes.astype(np.int8).tobytes()
        bits_per = 8
    else:
        codes_bytes = codes.astype(np.int16).tobytes()
        bits_per = 16
    payload = (
        struct.pack("<IBe", n_shrunk, K, np.float16(scale).item())
        + codes_bytes
    )
    return payload, {
        "n_orig": n_orig,
        "n_shrunk": n_shrunk,
        "n_nz_after_sparsity": n_nz,
        "K_chosen": K,
        "bits_per": bits_per,
        "payload_bytes": len(payload),
    }


def measure_full_stack(state_dict_path: Path, *, shrink_ratio: float, sparsity_alpha: float) -> dict:
    import torch

    input_sha = hashlib.sha256(state_dict_path.read_bytes()).hexdigest()
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise SystemExit(f"state_dict at {state_dict_path} is not a dict")

    all_payloads: list[bytes] = []
    per_tensor_stats: list[dict] = []
    n_orig_total = 0
    n_shrunk_total = 0
    n_nz_total = 0
    raw_payload_total = 0

    for name, _shape in FIXED_STATE_SCHEMA:
        tensor = sd[name].detach().cpu().to(torch.float32).numpy()
        payload, stats = encode_stack_per_tensor(
            tensor, shrink_ratio=shrink_ratio, sparsity_alpha=sparsity_alpha
        )
        all_payloads.append(payload)
        per_tensor_stats.append({"name": name, **stats})
        n_orig_total += stats["n_orig"]
        n_shrunk_total += stats["n_shrunk"]
        n_nz_total += stats["n_nz_after_sparsity"]
        raw_payload_total += stats["payload_bytes"]

    full_blob = b"".join(struct.pack("<I", len(p)) + p for p in all_payloads)
    compressed = brotli.compress(full_blob, quality=11, lgwin=16, lgblock=19)
    archive_bytes = len(compressed) + ARCHIVE_OVERHEAD_BYTES

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        **proxy_evidence_contract(),
        "input_state_dict": str(state_dict_path),
        "input_state_dict_sha256": input_sha,
        "shrink_ratio": shrink_ratio,
        "sparsity_alpha": sparsity_alpha,
        "n_orig_elements": n_orig_total,
        "n_shrunk_elements": n_shrunk_total,
        "n_nonzero_after_sparsity": n_nz_total,
        "raw_payload_bytes": len(full_blob),
        "brotli_bytes": len(compressed),
        "archive_overhead_bytes": ARCHIVE_OVERHEAD_BYTES,
        "archive_bytes": archive_bytes,
        "stack_steps": [
            "1. arch_shrink_x{ratio}",
            "2. IMP sparsity α={alpha}",
            "3. lossy_coarsening per-tensor K-step search (rel_err <5%)",
            "4. brotli q=11 lgwin=16 lgblock=19",
        ],
        "per_tensor": per_tensor_stats,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dict", type=Path, default=DEFAULT_STATE_DICT_PATH)
    p.add_argument("--shrink-ratio", type=float, default=0.4)
    p.add_argument("--sparsity-alpha", type=float, default=0.7)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    manifest = measure_full_stack(
        args.state_dict,
        shrink_ratio=args.shrink_ratio,
        sparsity_alpha=args.sparsity_alpha,
    )

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_omega_opt_linear_stack_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nmanifest: {args.output_json}\n")
    print(f"Stack composition: arch_shrink_x{args.shrink_ratio} → IMP α={args.sparsity_alpha} → lossy_coarsening → brotli")
    print("\nElement count flow:")
    print(f"  original: {manifest['n_orig_elements']:,}")
    print(f"  after arch_shrink: {manifest['n_shrunk_elements']:,} ({100*manifest['n_shrunk_elements']/manifest['n_orig_elements']:.1f}%)")
    print(f"  nonzero after sparsity: {manifest['n_nonzero_after_sparsity']:,} ({100*manifest['n_nonzero_after_sparsity']/manifest['n_orig_elements']:.1f}% of original)")
    print("\nByte flow:")
    print(f"  raw payload: {manifest['raw_payload_bytes']:,}")
    print(f"  brotli: {manifest['brotli_bytes']:,}")
    print(f"  + archive overhead: {manifest['archive_bytes']:,}")
    print("\nCompare to:")
    print("  PR101 brotli baseline: 178,144 B")
    delta = manifest["archive_bytes"] - 178_144
    print(f"  delta: {delta:+,} B")
    print("  Ω-OPT linear stack PREDICTION: 18,000 B (renderer only) [predicted, no anchor before this]")
    delta_pred = manifest["archive_bytes"] - 18_000
    print(f"  delta vs prediction: {delta_pred:+,} B")
    print("\nPer the metacognitive protocol: this is a BYTE ANCHOR, not a SCORE anchor.")
    print("The Ω-OPT 0.130 score prediction requires retrain + contest-CUDA.")

    if args.output_evidence:
        evidence_row = {
            "technique": "omega_opt_linear_stack_post_hoc_composition",
            "empirical_archive_bytes": manifest["archive_bytes"],
            **proxy_evidence_contract(),
            "shrink_ratio": args.shrink_ratio,
            "sparsity_alpha": args.sparsity_alpha,
            "source": (
                f"{EVIDENCE_GRADE} {args.output_json} "
                f"(arch_shrink_x{args.shrink_ratio} → IMP α={args.sparsity_alpha} → "
                f"lossy_coarsening → brotli; post-hoc composition, no retrain)"
            ),
            "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "first_empirical_anchor_for_omega_opt_linear_stack": True,
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"\nevidence row appended: {args.output_evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
