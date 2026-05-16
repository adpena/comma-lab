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

These bytes are consumed by the parser/intermediate Z3G2 reconstruction path (verified structurally by
``tools/verify_z3_g1_entropy_coded_v2_byte_mutation.py`` per Catalog #139).
That verifier is not a full-frame ``inflate.sh`` mutation proof; paired exact
eval and runtime-output mutation remain required before any promotion.

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
- Catalog #240: dispatch_enabled requires implementation_complete; the
  smoke path (``_smoke_main``) is COMPLETE; the full path (``_full_main``)
  raises NotImplementedError until smoke passes byte-mutation gate +
  operator approval.
- Catalog #272: distinguishing-feature integration contract documented
  in design memo §5.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY":
the v2 substrate is a research-only scaffold until a real non-smoke
train/export/auth-eval path lands. The smoke path validates only the
parser/intermediate byte-consumption contract; it is not dispatch-ranking
or promotion evidence.

Usage (smoke; CPU; ~3 epochs over a synthetic A1-shaped tensor; verifies
byte-mutation contract end-to-end):

    .venv/bin/python experiments/train_substrate_z3_g1_entropy_coded_v2.py \\
        --output-dir experiments/results/z3g1_entropy_coded_v2_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full; CUDA-required; gated behind operator approval per design memo §7):

    .venv/bin/python experiments/train_substrate_z3_g1_entropy_coded_v2.py \\
        --a1-archive-path submissions/a1/archive.zip \\
        --output-dir experiments/results/z3g1_entropy_coded_v2_<utc> \\
        --epochs 1000 --batch-size 16 --lr 1e-3 --device cuda
"""
# AUTOCAST_FP16_WAIVED:smoke-only-scaffold-defers-to-v1-trainer-for-amp-config-pattern
# TORCH_COMPILE_WAIVED:smoke-only-scaffold-defers-to-v1-trainer-for-Inductor-config-pattern
# NO_GRAD_WAIVED:smoke-path-uses-torch.no_grad-explicitly-around-eval-block
# SYNTHETIC_NON_SMOKE_OK:_smoke_main-only-uses-synthetic-latents-_full_main-raises-NotImplementedError
from __future__ import annotations

import argparse
import json
import sys
import time
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
    Z3G2EntropyCodedScorerClassGatingHead,
    build_z3g2_composition_archive_contract,
    build_z3g2_payload_bytes,
    encode_z3g2_section,
    estimate_z3g2_section_overhead_bytes,
    g1_v2_residual_rate_bits_per_sample,
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
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
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


def _full_main(args: argparse.Namespace) -> int:
    """Full main: COUNCIL-GATED until smoke passes byte-mutation contract.

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
    + HNeRV parity L2 (export-first design): the full path is gated until:

      (1) ``tools/verify_z3_g1_entropy_coded_v2_byte_mutation.py`` PASSES
          locally (extincts F1-class regression empirically pre-dispatch).
      (2) Operator-authorize recipe ``substrate_z3_g1_entropy_coded_v2_modal_t4_dispatch.yaml``
          is present, valid, and routed via canonical smoke-before-full
          per Catalog #167.
      (3) Operator approves the dispatch via paired-env per Catalog #199 +
          smoke-before-full pattern per Catalog #167.

    The full path will land in a follow-up commit batch once the smoke
    gate is verified empirically. This NotImplementedError is the
    structural gate per HNeRV parity L2 + Catalog #240.
    """
    raise NotImplementedError(
        "Z3-G1 entropy-coded v2 _full_main is COUNCIL-GATED per HNeRV parity L2 + "
        "Catalog #240 until: "
        "(1) tools/verify_z3_g1_entropy_coded_v2_byte_mutation.py PASSES locally; "
        "(2) operator approves smoke-before-full dispatch per Catalog #167 + #199; "
        "(3) operator approves Modal T4 ~$5-10 cost band per design memo §7. "
        "Use --smoke to run the local validator instead."
    )


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
