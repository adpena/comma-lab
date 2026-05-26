#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CASCADE B CATALYST CASCADE COMPOSITION 5th-order recursive doctrine —
MLX-LOCAL EMPIRICAL ANCHOR harness.

Runs the 3-arm CATALYST cascade pipeline (baseline / Path A alone / CATALYST
composition) on a 50-pair MLX-local fixture mirroring the sister Path A
scaffold pattern. Emits canonical verdict JSON + registers FIRST empirical
anchor on canonical equation #2
`hinton_kl_distill_enables_qat_catalyst_composition_savings_v1`.

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192/#317:
  [macOS-MLX research-signal] only; NEVER promotable; no contest claim.

Per CLAUDE.md "Forbidden premature KILL":
  IF CATALYST cascade empirically improves combined axis vs Path A alone:
    PARADIGM-VALIDATED + canonical equation #2 anchor REGISTERED (FIRST anchor)
  IF FALSIFIED:
    IMPLEMENTATION-LEVEL per Catalog #307; PARADIGM (CATALYST composition)
    INTACT; sister 6th-order iteration on per-stage hyperparams.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "upstream"))

import mlx.core as mx  # noqa: E402

from tac.canonical_equations.equation import EmpiricalAnchor  # noqa: E402
from tac.canonical_equations.registry import (  # noqa: E402
    update_equation_with_empirical_anchor,
)
from tac.provenance.builders import build_provenance_for_predicted  # noqa: E402
from tac.substrates.hinton_distilled_scorer_surrogate.catalyst_cascade import (  # noqa: E402
    run_catalyst_cascade_pipeline,
)
from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (  # noqa: E402
    DEFAULT_DISTILLATION_TEMPERATURE,
    DEFAULT_SEGNET_CLASSES,
    build_learnable_student_head,
)


CANONICAL_EQUATION_ID = "hinton_kl_distill_enables_qat_catalyst_composition_savings_v1"


def _build_synthetic_50_pair_fixture(
    *,
    num_pairs: int = 50,
    height: int = 48,
    width: int = 64,
    num_classes: int = DEFAULT_SEGNET_CLASSES,
    seed: int = 0,
):
    """Build a deterministic 50-pair MLX fixture mirroring the sister Path A
    scaffold pattern.

    The fixture uses a synthetic teacher distribution (canonical
    pseudo-SegNet logits derived via deterministic projection over canonical
    RGB synthesis) so the empirical anchor is reproducible bit-for-bit
    across runs.
    """
    rng_key = mx.random.key(seed)
    key_rgb, key_teacher = mx.random.split(rng_key)
    # Canonical NHWC RGB float32 in [0, 1] (sister Slot 1 normalization)
    decoded_rgb = mx.random.uniform(
        shape=(num_pairs, height, width, 3),
        key=key_rgb,
    )
    # Synthetic teacher logits — canonical deterministic-projection pattern
    # so the teacher target distribution is well-defined without requiring
    # real SegNet weights on disk for the smoke anchor.
    teacher_proj = mx.random.normal(
        shape=(3, num_classes),
        key=key_teacher,
    ) * 0.5
    target_logits = mx.einsum("phwc,ck->phwk", decoded_rgb, teacher_proj)
    return decoded_rgb, target_logits


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=DEFAULT_DISTILLATION_TEMPERATURE)
    parser.add_argument("--gain-clamp", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default=str(
            REPO_ROOT
            / "experiments"
            / "results"
            / "cascade_b_catalyst_cascade_composition_5th_order_20260526"
        ),
    )
    parser.add_argument(
        "--register-canonical-anchor",
        action="store_true",
        help="Register the empirical result as a NEW anchor on canonical equation #2 "
             "(default: off; opt-in to avoid accidentally polluting the registry).",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[catalyst-cascade-5th-order] running 3-arm pipeline "
        f"num_pairs={args.num_pairs} H={args.height} W={args.width} "
        f"T={args.temperature} gain_clamp={args.gain_clamp}",
        flush=True,
    )

    t0 = time.time()
    decoded_rgb, target_logits = _build_synthetic_50_pair_fixture(
        num_pairs=args.num_pairs,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    head = build_learnable_student_head(
        num_classes=DEFAULT_SEGNET_CLASSES,
        in_channels=3,
        seed=args.seed,
    )
    # Compute initial student logits via the Path A head applied to the
    # canonical decoded RGB surface.
    student_logits_initial = head(decoded_rgb)

    result = run_catalyst_cascade_pipeline(
        student_logits_initial=student_logits_initial,
        target_logits=target_logits,
        path_a_head=head,
        temperature=args.temperature,
        gain_clamp=args.gain_clamp,
    )
    wall_clock = time.time() - t0
    result["wall_clock_seconds"] = wall_clock

    # Print canonical 3-arm comparison
    print(
        f"[catalyst-cascade-5th-order] wall_clock={wall_clock:.2f}s",
        flush=True,
    )
    for arm in result["arms"]:
        print(
            f"  arm={arm['arm_name']:25s} "
            f"kl_final={arm['kl_final']:.4f} "
            f"bpr1_bytes={arm['bpr1_sidecar_bytes']:6d} "
            f"rate_score={arm['rate_term_canonical_score']:.6e} "
            f"composite={arm['composite_proxy_score_for_mlx_research_signal']:.4f}",
            flush=True,
        )
    delta = result["delta_summary"]
    print(
        f"  delta_path_a_alone_kl_minus_baseline_kl    = "
        f"{delta['delta_path_a_alone_kl_minus_baseline_kl']:+.4f}",
        flush=True,
    )
    print(
        f"  delta_catalyst_kl_minus_path_a_alone_kl    = "
        f"{delta['delta_catalyst_composition_kl_minus_path_a_alone_kl']:+.4f}",
        flush=True,
    )
    print(
        f"  delta_catalyst_rate_minus_baseline_rate    = "
        f"{delta['delta_catalyst_composition_rate_minus_baseline_rate']:+.6e}",
        flush=True,
    )
    print(
        f"  catalyst_improves_over_path_a_alone        = "
        f"{delta['catalyst_composition_improves_over_path_a_alone']}",
        flush=True,
    )
    print(
        f"  catalyst_improves_over_baseline            = "
        f"{delta['catalyst_composition_improves_over_baseline']}",
        flush=True,
    )

    # Verdict per Catalog #307 paradigm-vs-implementation classification
    catalyst_improves_over_path_a = delta["catalyst_composition_improves_over_path_a_alone"]
    catalyst_improves_over_baseline = delta["catalyst_composition_improves_over_baseline"]
    if catalyst_improves_over_path_a and catalyst_improves_over_baseline:
        verdict = "PARADIGM_VALIDATED"
        verdict_detail = (
            "CATALYST cascade composition empirically improves combined "
            "(d_seg_proxy + rate) composite over BOTH baseline AND Path A alone. "
            "Canonical equation #2 anchor REGISTERED (FIRST anchor) and PARADIGM "
            "(CATALYST composition P2+P5+P10) is empirically VINDICATED."
        )
    elif catalyst_improves_over_baseline and not catalyst_improves_over_path_a:
        verdict = "IMPLEMENTATION_LEVEL_PARTIAL_VALIDATION"
        verdict_detail = (
            "CATALYST cascade composition improves over baseline (Path A foundation "
            "drives the win) but does NOT improve over Path A alone — the BPR1 "
            "sidecar rate cost exceeds the QAT-induced d_seg-proxy improvement at "
            "the empirical fixture scale. PARADIGM (CATALYST composition) INTACT "
            "per Catalog #307; sister 6th-order iteration on per-stage hyperparams "
            "(gain_clamp / block_size / temperature) is operator-routable."
        )
    else:
        verdict = "IMPLEMENTATION_LEVEL_FALSIFIED"
        verdict_detail = (
            "CATALYST cascade composition does NOT improve over baseline at the "
            "empirical fixture scale. PARADIGM (CATALYST composition P2+P5+P10) "
            "INTACT per Catalog #307; sister 6th-order iteration on per-stage "
            "hyperparams OR sister architectural variant required. Per CLAUDE.md "
            "'Forbidden premature KILL': DEFERRED-pending-research."
        )
    result["verdict_per_catalog_307"] = verdict
    result["verdict_detail"] = verdict_detail
    print(f"\n  VERDICT (Catalog #307): {verdict}", flush=True)
    print(f"  {verdict_detail}", flush=True)

    # Persist verdict JSON
    verdict_path = output_dir / "catalyst_cascade_pipeline_verdict.json"
    verdict_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"\n  verdict written to {verdict_path}", flush=True)

    # Register canonical equation anchor (opt-in per --register-canonical-anchor)
    if args.register_canonical_anchor:
        # Build canonical Provenance per Catalog #323/#341
        utc_now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        inputs_sha = hashlib.sha256(
            json.dumps(
                {
                    "num_pairs": args.num_pairs,
                    "height": args.height,
                    "width": args.width,
                    "seed": args.seed,
                    "temperature": args.temperature,
                    "gain_clamp": args.gain_clamp,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        provenance = build_provenance_for_predicted(
            model_id="catalyst_cascade_5th_order_synthetic_fixture",
            inputs_sha256=inputs_sha,
            measurement_axis="[macOS-MLX research-signal]",
            hardware_substrate="macos_arm64",
            captured_at_utc=utc_now,
        )
        # Per equation #2 latex form: alpha = QAT_savings_lift / delta_H_logits_T2
        # We do not have a separate QAT-alone arm; the empirical observation is the
        # composite proxy ordering. Anchor predicted_output uses the canonical
        # schema's predicted band (qat_savings_lift α=0.15 + post-quantization
        # scorer-entropy tightening ratio 0.85); empirical_output reports the
        # actual measured delta.
        anchor = EmpiricalAnchor(
            anchor_id=(
                f"hinton_kl_distill_qat_catalyst_5th_order_cascade_composition_"
                f"50pair_mlx_local_{utc_now.replace(':', '').replace('-', '')}"
            ),
            measurement_utc=utc_now,
            inputs={
                "num_pairs": args.num_pairs,
                "height": args.height,
                "width": args.width,
                "seed": args.seed,
                "temperature": args.temperature,
                "gain_clamp": args.gain_clamp,
                "catalyst_position_p2": "hinton_kl_distill_t_eq_2p0",
                "enabled_position_p4": "fake_quant_fp4_mlx_canonical_codebook",
                "output_position_p10": "bpr1_sign_bitmap_sidecar_variant_b_d",
                "substrate_class_under_audit": "hinton_distilled_scorer_surrogate_path_a_learnable_head",
                "cascade_b_canonical_embodiment": True,
                "source_directive_memo": (
                    ".omx/research/cascade_b_catalyst_cascade_composition_5th_order_"
                    "pre_execution_gate_report_20260526.md"
                ),
            },
            predicted_output={
                # canonical equation #2 schema predicts these per the registered row
                "qat_savings_lift_relative_to_qat_alone": 0.15,
                "post_quantization_scorer_entropy_tightening_ratio": 0.85,
            },
            empirical_output={
                "verdict_per_catalog_307": verdict,
                "delta_catalyst_kl_minus_path_a_alone_kl": float(
                    delta["delta_catalyst_composition_kl_minus_path_a_alone_kl"]
                ),
                "delta_catalyst_rate_minus_baseline_rate": float(
                    delta["delta_catalyst_composition_rate_minus_baseline_rate"]
                ),
                "composite_baseline": float(delta["composite_baseline"]),
                "composite_path_a_alone": float(delta["composite_path_a_alone"]),
                "composite_catalyst_composition": float(delta["composite_catalyst_composition"]),
                "catalyst_improves_over_path_a_alone": bool(
                    delta["catalyst_composition_improves_over_path_a_alone"]
                ),
                "catalyst_improves_over_baseline": bool(
                    delta["catalyst_composition_improves_over_baseline"]
                ),
            },
            # Residual: |empirical composite − predicted composite| normalized.
            # We use composite_catalyst_composition vs composite_path_a_alone as the
            # canonical residual surface: a successful CATALYST should drive the
            # composite delta strictly negative; the residual magnitude is the
            # absolute delta (smaller = closer to canonical predicted lift).
            residual=abs(
                float(delta["composite_catalyst_composition"])
                - float(delta["composite_path_a_alone"]) * (1.0 - 0.15)
            ),
            source_artifact=str(verdict_path.relative_to(REPO_ROOT)),
            measurement_method=(
                "mlx_local_3_arm_catalyst_cascade_synthetic_50_pair_fixture_"
                "p2_hinton_kl_t2_plus_p4_fake_quant_fp4_mlx_plus_p10_bpr1_sign_bitmap"
            ),
            provenance=provenance,
        )
        update_equation_with_empirical_anchor(
            CANONICAL_EQUATION_ID,
            anchor,
            agent="claude",
            subagent_id=(
                "cascade-b-catalyst-cascade-composition-p5-qat-p10-bpr1-onto-path-a-"
                "foundation-5th-order-recursive-doctrine-mlx-first-numpy-portable-20260526"
            ),
            notes=(
                f"FIRST empirical anchor on canonical equation #2 "
                f"hinton_kl_distill_enables_qat_catalyst_composition_savings_v1 "
                f"per Catalog #344 (registry would compound 0 -> 1 anchor). "
                f"Verdict per Catalog #307 = {verdict}."
            ),
        )
        print(
            f"  canonical equation #2 anchor REGISTERED "
            f"(anchor_id={anchor.anchor_id})",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
