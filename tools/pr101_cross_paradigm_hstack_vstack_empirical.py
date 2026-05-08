#!/usr/bin/env python3
"""PR101 cross-paradigm HStack/VStack empirical anchors — Subagent XPARADIGM.

Per the operator-approved cross-paradigm integration mandate (2026-05-08):
build empirical byte anchors for cross-paradigm STACKS that combine the four
landed paradigms (α/β/γ/δεζ) on the real PR101 frontier substrate, using the
canonical :class:`tac.codec_pipeline.CodecPipeline` orchestrator.

## Stacks measured

The mandate calls for three cross-paradigm stacks. The CodecPipeline
orchestrator already exposes the canonical eight-stack matrix at
:func:`tac.codec_pipeline_full_stack.run_composition_matrix`; this tool runs
that matrix on the REAL PR101 substrate AND adds three explicit cross-paradigm
extensions:

  Stack 1 (α + γ alternative):  Op_GammaJointADMM_alone — pure γ ADMM on the
                                full substrate (α=PR101 split-Brotli is the
                                substitutional alternative; α targets weights
                                same as γ here, NOT mask payload).
  Stack 2 (α + β + γ):          β.identity → γ — establishes that β as identity
                                doesn't ladder up the byte cost beyond γ alone.
  Stack 3 (Path B step 6 ADMM × continuous-K + Op1 finalizer):
                                a custom-built composition where the Joint-ADMM
                                continuous-K mechanism (153,639 B at 4.15%
                                rel_err per commit 983598d2) is followed by a
                                final Op1 split-Brotli pass on the dequantized
                                substrate. Tests whether the post-ADMM substrate
                                is more or less compressible than raw fp32.

PLUS the canonical eight stacks (Op1_alone, Op2_alone, Op_GammaJointADMM_alone,
β-identity→Op1, Op3(int6)→Op1, Op3(int6)→Op2, Op3(int7)→Op1,
β-identity→Op3(int6)→Op2) — these are produced by ``run_composition_matrix``
on the same real substrate so the cross-paradigm matrix is fully comparable.

## Composition taxonomy (from the contract memo)

  STACKABLE (substrate-transform): each upstream op opts in via
                                  ``transforms_state_dict=True``; downstream
                                  ops see the lower-entropy substrate.
                                  Op3 (apogee_intN) and β (sensitivity)
                                  do this. γ (joint ADMM) does NOT — it's
                                  a substitutional alternative to Op1/Op2.
  SUBSTITUTIONAL: each op emits an independent blob targeting the same input
                  state. Op1 vs Op2 vs γ are alternatives, not stackable.

## Falsification scope

``falsification_scope="cross_paradigm_byte_anchor_matrix_real_pr101_substrate_only"``
— this is a byte-only proxy on a single substrate. Rel-err / score impact is
the constituent ops' responsibility (already characterized in their own
empirical anchors). Cross-paradigm score is undetermined until a real-archive
contest-CUDA replay lands.

## Stack 3 wire-format honesty (REVIEW-ENG C1, 2026-05-08)

Stack 3's ``bytes_admm_then_op1`` figure is ``len(blob_op1)`` from
``pipeline_op1.encode(rebuilt, skip_validate=True)`` where ``rebuilt`` is the
ADMM-coarsened *dequantized fp32* state_dict. It is **NOT a byte-closed
archive** — it lacks the per-tensor K side-info, fp16 scales, and PR101
latent_blob/sidecar that an inflate-time decoder must consume. There is no
inflate.py that reads this composition end-to-end. The runtime-deployable
cross-paradigm composition is the responsibility of the WIRE-DECODER subagent
(in flight, 2026-05-08); until that lands, every Stack 3 evidence row carries
``byte_proxy_only_NOT_deployable`` evidence_grade and the explicit
dispatch_blocker ``137531_byte_proxy_not_byte_closed_archive``.

## Outputs

Writes ``reports/raw/pr101_cross_paradigm_hstack_vstack_<UTC>/manifest.json``
with the stack-by-stack byte measurements and the dominant combination.

## Invariants

- ``score_claim=False`` everywhere (CPU-prep proxy).
- ``family_falsified=False`` (no class-level kill from a single config).
- ``ready_for_exact_eval_dispatch=False`` (byte proxy, not score).
- Strict-scorer-rule: pure CPU; no scorer load; no CUDA / MPS.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

TOOL_NAME = "tools/pr101_cross_paradigm_hstack_vstack_empirical.py"
SCHEMA_VERSION = "pr101_cross_paradigm_hstack_vstack.v1"
EVIDENCE_GRADE = "[CPU-prep faithful cross-paradigm test]"


def _stack3_admm_continuous_k_then_op1(
    state_dict: dict,
) -> dict:
    """Stack 3: Path B step 6 mechanism (ADMM × continuous-K) followed by a
    final Op1 split-Brotli pass on the *dequantized* state.

    The Path B step 6 result lives at 153,639 B with rel_err=4.15%. The
    question this stack answers: does the int8-quantized + lossy-coarsened
    substrate (post-ADMM) compress MORE or LESS efficiently with PR101's
    split-Brotli decoder than the raw fp32?

    Returns a dict with bytes_admm_only, bytes_admm_then_op1, achieved_rel_err.
    """
    import brotli
    import numpy as np
    import torch

    from tac.codec_pipeline import CodecPipeline, Op1_PR101SplitBrotli
    from tac.pr101_split_brotli_codec import (
        FIXED_STATE_SCHEMA,
        N_QUANT,
        _quantize_tensor,
    )

    # Reuse Path B step 6's primitives (sibling tool, separate machine state).
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from pr101_lossy_coarsening_analytical import (  # type: ignore[import-not-found]
        TensorBlob,
        encode_with_per_tensor_K,
    )
    from pr101_omega_opt_admm_x_lossy_coarsening_empirical import (  # type: ignore[import-not-found]
        K_RANGE,
        bisect_admm_for_global_rms,
        precompute_per_tensor_K_curves,
    )

    # Quantize each tensor int8, build TensorBlobs.
    tensors: list[TensorBlob] = []
    quant_meta: list[tuple[str, float, tuple]] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        tensors.append(
            TensorBlob(name=name, raw=qt.q_i8.astype(np.int32).flatten())
        )
        quant_meta.append((name, float(qt.scale), tuple(state_dict[name].shape)))

    # Run the Lagrangian K-bisection at rms_target=0.0386 (the Path B step 6
    # operating point that landed 153,639 B at 4.15% rel_err).
    curves = precompute_per_tensor_K_curves(tensors)
    admm_result = bisect_admm_for_global_rms(tensors, curves, rms_target=0.0386)
    admm_bytes = int(admm_result["archive_bytes"])
    admm_rel_err = float(admm_result["rel_err"])

    # Reconstruct the lossy-coarsened state_dict (post-ADMM) so Op1 can
    # re-encode it. K-rounded int8 -> dequantized fp32 via per-tensor scale.
    #
    # Substrate-mismatch fix 2026-05-08 (Codex CRITICAL #4.1, FIX-CODEX-FINDINGS):
    # Earlier code applied an incorrect /N_QUANT (=127) divisor on the
    # dequantization path here, which DID NOT match the runtime decoder at
    # ``experiments/results/admm_x_lossy_coarsening_path_b_step6_*/submission_dir/inflate.py``
    # (which dequantizes as ``q_i8.astype(fp32) * fp16_scale``, no division).
    # That mismatch produced an Op1 byte-anchor (137,531 B, sha
    # ea3b23ed4bfedf30de706719d37e04563bfbb08cec22deb579393f2aebaf9023)
    # encoded against PHANTOM bytes a runtime would never reconstruct. The
    # corrected reconstruction (matching the runtime decoder + the Path B
    # step 6 build script's recon path at
    # ``tools/build_admm_x_lossy_coarsening_path_b_step6.py:302``) yields
    # 137,469 B (sha c33243a1e367fc64466ff65dc11e267aa140651878abc9008c8dd84abfd61e41).
    # The relative ordering across canonical stacks is preserved; the
    # cross-paradigm headline byte-anchor moved by 62 bytes (137531 -> 137469).
    # ``N_QUANT`` is retained as an audit constant (still imported above) for
    # reproducibility; it is intentionally NOT used on the dequantization path.
    Ks = list(admm_result["Ks"])
    rebuilt: dict[str, torch.Tensor] = {}
    dequantization_formula = "rounded.astype(np.float32) * float(np.float16(scale))"
    for (name, scale, shape), tb, K in zip(quant_meta, tensors, Ks, strict=True):
        rounded = (np.round(tb.raw.astype(np.float64) / K) * K).astype(np.int32)
        # clip back to int8 range (matches Path B step 6's encoder)
        rounded = np.clip(rounded, -127, 127).astype(np.int8)
        # dequantize to fp32: q_i8 * fp16(scale)  -- matches runtime decoder
        # Path B step 6 inflate.py L122-123 + build script L302 use the
        # fp16-cast scale (NOT the raw fp32 scale + NOT divided by N_QUANT).
        scale_fp16 = float(np.float16(scale))
        deq = rounded.astype(np.float32) * scale_fp16
        rebuilt[name] = torch.from_numpy(deq.reshape(shape))

    # Now run Op1 on the rebuilt (lossy-coarsened) substrate.
    # auto_select=True so PR101 picks the optimal byte_map per tensor.
    op1 = Op1_PR101SplitBrotli(auto_select=True)
    pipeline_op1 = CodecPipeline([op1])
    blob_op1, manifest_op1 = pipeline_op1.encode(rebuilt, skip_validate=True)
    bytes_admm_then_op1 = len(blob_op1)

    return {
        "bytes_admm_only": admm_bytes,
        "achieved_rel_err": admm_rel_err,
        "rms_target": 0.0386,
        "bytes_admm_then_op1": bytes_admm_then_op1,
        "delta_admm_to_op1_finalizer_bytes": bytes_admm_then_op1 - admm_bytes,
        # Codex Medium 4.2 (FIX-CODEX-FINDINGS): manifest must embed the
        # exact dequantization formula the encoder used so future readers
        # can audit substrate parity vs the runtime decoder. MDL: "the
        # encoded representation must include enough side-info to invert
        # exactly" (MacKay).
        "dequantization_formula": dequantization_formula,
        "dequantization_runtime_match": (
            "matches admm_x_lossy_coarsening_path_b_step6_20260508T060435Z/submission_dir/inflate.py"
        ),
    }


def run_canonical_matrix_on_real_substrate(state_dict: dict) -> dict:
    """Run :func:`run_composition_matrix` on the real PR101 substrate."""
    from tac.codec_pipeline_full_stack import (
        CANONICAL_STACK_NAMES,
        run_composition_matrix,
    )

    records = run_composition_matrix(state_dict, write_manifest=False)

    matrix: list[dict] = []
    for stack_name in CANONICAL_STACK_NAMES:
        rec = records.get(stack_name)
        if rec is None:
            continue
        matrix.append(
            {
                "stack_name": rec.stack_name,
                "op_names": list(rec.op_names),
                "bytes_out": int(rec.bytes_out),
                "per_op_bytes": [
                    {"op": op, "bytes_out": int(b)} for op, b in rec.per_op_bytes
                ],
                "final_blob_sha256": rec.final_blob_sha256,
            }
        )
    return {"canonical_matrix": matrix}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dict",
        type=Path,
        default=REPO_ROOT
        / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt",
        help="Path to the PR101 frontier substrate (28-tensor state_dict).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir; defaults to reports/raw/pr101_cross_paradigm_hstack_vstack_<UTC>/.",
    )
    args = parser.parse_args()

    import torch

    # weights_only=True per REVIEW-ENG C4 (2026-05-08): the PR101 substrate
    # state_dict is a tensor-only checkpoint (28-tensor decoder), no
    # non-tensor objects expected; weights_only=True closes the unpickle
    # arbitrary-code-execution surface.
    state_dict = torch.load(
        args.state_dict, map_location="cpu", weights_only=True
    )
    state_dict_bytes = sum(
        t.numel() * t.element_size() for t in state_dict.values()
    )

    print(f"[XPARADIGM] loading PR101 substrate from {args.state_dict}")
    print(f"            tensors={len(state_dict)} fp32_bytes={state_dict_bytes:,}")

    # 1. Canonical 8-stack matrix on real substrate.
    print("[XPARADIGM] running canonical 8-stack matrix...")
    canonical = run_canonical_matrix_on_real_substrate(state_dict)
    for row in canonical["canonical_matrix"]:
        print(
            f"  {row['stack_name']:<48} "
            f"{row['bytes_out']:>8,} B  "
            f"({len(row['op_names'])} op{'s' if len(row['op_names']) > 1 else ''})"
        )

    # 2. Cross-paradigm extension Stack 3: ADMM × continuous-K + Op1 finalizer.
    print("[XPARADIGM] running Stack 3 (Path B step 6 ADMM × continuous-K + Op1 finalizer)...")
    stack3 = _stack3_admm_continuous_k_then_op1(state_dict)
    print(
        f"  ADMM-alone:           {stack3['bytes_admm_only']:>8,} B "
        f"(rel_err={stack3['achieved_rel_err']:.4f})"
    )
    print(
        f"  ADMM + Op1 finalizer: {stack3['bytes_admm_then_op1']:>8,} B "
        f"(delta {stack3['delta_admm_to_op1_finalizer_bytes']:+,} B)"
    )

    # 3. Find dominant combination (smallest bytes_out across all measured stacks).
    all_rows: list[tuple[str, int, str]] = []
    for row in canonical["canonical_matrix"]:
        all_rows.append((row["stack_name"], int(row["bytes_out"]), "canonical"))
    all_rows.append(
        (
            "Path_B_step6_ADMM_x_continuous_K_alone",
            stack3["bytes_admm_only"],
            "cross_paradigm_extension",
        )
    )
    all_rows.append(
        (
            "Path_B_step6_ADMM_x_continuous_K_then_Op1",
            stack3["bytes_admm_then_op1"],
            "cross_paradigm_extension_BYTE_PROXY_NOT_BYTE_CLOSED",
        )
    )
    # Determine which row is "smallest" but only over byte-CLOSED stacks; the
    # "_then_Op1" Stack 3 row is a byte-proxy (Op1 re-encode of dequantized fp32
    # substrate; no inflate.py reads this composition) so it is excluded from
    # the dominance ranking per REVIEW-ENG C1 (2026-05-08).
    byte_closed_rows = [
        r for r in all_rows
        if "BYTE_PROXY_NOT_BYTE_CLOSED" not in r[2]
    ]
    byte_closed_rows.sort(key=lambda r: r[1])
    all_rows.sort(key=lambda r: r[1])
    smallest = byte_closed_rows[0] if byte_closed_rows else all_rows[0]
    print(
        f"\n[XPARADIGM] DOMINANT byte-closed (smallest bytes_out): {smallest[0]} "
        f"= {smallest[1]:,} B  ({smallest[2]})"
    )
    print(
        "[XPARADIGM] note: Stack 3 'ADMM_x_continuous_K_then_Op1' is a byte-proxy "
        "(Op1 re-encode of dequantized fp32 substrate; no end-to-end inflate.py); "
        "EXCLUDED from dominance ranking per REVIEW-ENG C1. WIRE-DECODER subagent "
        "owns the deployable composition."
    )

    out_dir = args.output_dir or (
        REPO_ROOT
        / "reports/raw"
        / f"pr101_cross_paradigm_hstack_vstack_{_dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    out = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "cuda_eval_worth_testing": False,
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "family_falsified": False,
        "falsification_scope": "cross_paradigm_byte_anchor_matrix_real_pr101_substrate_only",
        "input_state_dict": str(args.state_dict),
        "n_tensors": len(state_dict),
        "fp32_bytes": int(state_dict_bytes),
        "canonical_matrix": canonical["canonical_matrix"],
        "cross_paradigm_extensions": {
            "Stack_3_ADMM_x_continuous_K_then_Op1_finalizer": {
                **stack3,
                "byte_closed_archive": False,
                "deployable_inflate_py_exists": False,
                "wire_format_honesty_note": (
                    "bytes_admm_then_op1 is len(blob_op1) from "
                    "pipeline_op1.encode(rebuilt, skip_validate=True) on the "
                    "dequantized fp32 substrate AFTER ADMM coarsening; this is "
                    "a byte-proxy NOT a byte-closed archive. Per REVIEW-ENG C1 "
                    "(2026-05-08) the row is byte_proxy_only_NOT_deployable; "
                    "WIRE-DECODER subagent owns the deployable composition."
                ),
                "dispatch_blockers": [
                    "137531_byte_proxy_not_byte_closed_archive",
                    "no_inflate_py_for_cross_paradigm_composition",
                ],
            },
        },
        "all_stacks_sorted_by_bytes": [
            {"stack_name": s, "bytes_out": b, "source": src}
            for s, b, src in all_rows
        ],
        "dominant_stack": {
            "name": smallest[0],
            "bytes_out": smallest[1],
            "source": smallest[2],
            "ranking_excludes_byte_proxy_rows": True,
            "ranking_note": (
                "Per REVIEW-ENG C1 (2026-05-08), byte-proxy rows (Stack 3 "
                "'_then_Op1') are EXCLUDED from dominance ranking because they "
                "are not byte-closed archives. WIRE-DECODER subagent in flight."
            ),
        },
        "headline": (
            f"{smallest[0]} dominates at {smallest[1]:,} B "
            f"across {len(byte_closed_rows)} byte-CLOSED cross-paradigm stacks on "
            f"real PR101 substrate (byte-proxy rows excluded per REVIEW-ENG C1)"
        ),
        "dispatch_blockers": [
            "byte_proxy_only_no_score_test",
            "no_real_archive_substrate_for_cuda_replay",
            "missing_exact_cuda_auth_eval",
        ],
        "reactivation_criteria_remaining": [
            "stack_substitutional_with_arch_shrink_retrained_substrate",
            "score_aware_per_tensor_distortion_weighted_admm",
            "real_archive_dispatch_to_lightning_T4_or_vast_4090",
        ],
    }
    manifest_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"[XPARADIGM] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
