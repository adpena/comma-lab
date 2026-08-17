#!/usr/bin/env python3
"""Measure the d_seg cost of DROPPING FiLM rows from a trained semantic renderer.

WHY THIS EXISTS (the wrong-instrument correction it repairs)
------------------------------------------------------------
``--film-row-dropout`` uses INVERTED dropout: surviving rows are rescaled by
``1/(1-p)`` so the expectation is unchanged (``editability_levers._row_dropout``
says so in a comment). Inverted dropout therefore buys ROBUSTNESS TO ROW REMOVAL
— it does NOT drive any particular row's norm toward zero. Measuring row norms to
test the lever answers a question the mechanism never posed: MAIN did exactly
that on 2026-08-17 and found (correctly, and irrelevantly) that the norms barely
move. The quantity the lever is defined against is the DROPPABILITY of rows, and
droppability is only visible under ablation.

WHAT IT MEASURES
----------------
For each supplied checkpoint and each ``k`` on a ladder, zero the ``k``
lowest-ranked rows of every FiLM tensor and evaluate ``quantized_exact_seg``
through the trainer's own path.

Two referents are held deliberately, because a lever measured against a
different referent than it was built against is not measured at all:

* the RANKING is ``editability_levers.film_row_order`` — the same descending-L2
  order the lever protects with ``--film-row-dropout-protect-top`` and the same
  order the shipped keep-ladders prune with, so "drop the k lowest" IS "what
  keep-(rows-k) would discard";
* the INSTRUMENT is ``train_semantic_quantized_resumable._evaluate_semantic_pairs``
  on the checkpoint's ``state_dict`` (``deployment_weights == "ema_shadow"``) —
  the exact object and function whose numbers the training arms report, so the
  ``k=0`` row must reproduce the arm's own endpoint.

CONSUMERS
---------
* lever ``--film-row-dropout`` (#1092 drain, lever 1) — is the trained model
  cheaper to prune than its matched control?
* lever ``--film-row-dropout-protect-top`` (#1092 drain) — same curve with the
  protected head held out.
* ``ddm_mz2``'s retained-but-unscored FiLM-row sparsity candidates
  (-130..-2,051 B): those are byte counts with no measured d_seg price. This
  supplies the price.

PAYLOAD CUSTODY
---------------
The ablated weights are a deterministic function of (checkpoint sha256, tensor
name, k) through ``film_row_order``. The result JSON records the sha256 of every
checkpoint AND the exact dropped row indices per tensor per k, so any ablated
state is reconstructible byte-for-byte without storing a copy of each one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from safetensors.torch import load_file

from tac.pr130_lift.editability_levers import FILM_ROW_FAMILY, film_row_order
from tac.pr130_lift.train_semantic_quantized_resumable import (
    N_TOTAL_PAIRS,
    _evaluate_semantic_pairs,
    _load_lifted_qat,
)

TOTAL_PIXELS = 117_964_800


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _drop_plan(state_dict: dict[str, torch.Tensor], k: int) -> dict[str, list[int]]:
    """Rows each FiLM tensor would lose at keep-(rows-k), in the shipped order."""

    plan: dict[str, list[int]] = {}
    for name in sorted(FILM_ROW_FAMILY):
        value = state_dict.get(name)
        if value is None or value.ndim < 2:
            continue
        order = film_row_order(value)
        if k > len(order):
            raise ValueError(f"k={k} exceeds {name} row count {len(order)}")
        # film_row_order is DESCENDING by norm, so the k lowest are the tail.
        plan[name] = sorted(order[len(order) - k :]) if k else []
    return plan


def _apply_drop(
    state_dict: dict[str, torch.Tensor], plan: dict[str, list[int]]
) -> dict[str, torch.Tensor]:
    ablated = {key: value.clone() for key, value in state_dict.items()}
    for name, rows in plan.items():
        if rows:
            ablated[name][rows] = 0.0
    return ablated


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="labelled checkpoint, repeatable (e.g. EF3000=/path/ckpt.pt)",
    )
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--k-ladder",
        default="0,8,16,32,64,96",
        help="rows dropped per FiLM tensor; k=0 must reproduce the arm endpoint",
    )
    parser.add_argument(
        "--expect-k0",
        action="append",
        default=[],
        metavar="LABEL=D_SEG",
        help=(
            "the arm's own endpoint d_seg that k=0 MUST reproduce. This is the "
            "positive control that binds the harness to the arm's axis: device, "
            "eval batch size, bits, pair set and weight basis all have to match or "
            "the curve is measured beside the endpoint it is compared against. "
            "Compared EXACTLY -- the path is deterministic and was observed to "
            "reproduce EF3000 to all 17 digits, so any difference is a real "
            "instrument mismatch, not float noise."
        ),
    )
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="MUST match the producing run: batch shape is part of the instrument",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--pairs",
        type=int,
        default=N_TOTAL_PAIRS,
        help="pair count; the default is the full n600 population on purpose",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 1 <= args.pairs <= N_TOTAL_PAIRS:
        raise ValueError("--pairs must be in [1,600]")
    ladder = [int(token) for token in args.k_ladder.split(",") if token.strip()]

    arms: list[tuple[str, Path]] = []
    for spec in args.arm:
        label, _, raw_path = spec.partition("=")
        if not label or not raw_path:
            raise ValueError(f"--arm must be LABEL=PATH, got {spec!r}")
        if any(label == seen for seen, _ in arms):
            raise ValueError(f"--arm label {label!r} is repeated; labels key the rows")
        arms.append((label, Path(raw_path)))

    expect_k0: dict[str, float] = {}
    for spec in args.expect_k0:
        label, _, raw_value = spec.partition("=")
        if not label or not raw_value:
            raise ValueError(f"--expect-k0 must be LABEL=D_SEG, got {spec!r}")
        expect_k0[label] = float(raw_value)
    if expect_k0 and 0 not in ladder:
        raise ValueError("--expect-k0 given but k=0 is not on the ladder to check it")
    # A control that a typo can silently switch off is not a control: an
    # --expect-k0 label that matches no --arm would simply never be compared.
    unbound = sorted(set(expect_k0) - {label for label, _ in arms})
    if unbound:
        raise ValueError(
            f"--expect-k0 labels {unbound} match no --arm; the positive control "
            "would never fire"
        )

    # `modules` (SegNet + weights path) lives in the pinned upstream snapshot and
    # is imported the way the trainer imports it — via sys.path, never vendored.
    root = args.challenge_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import modules  # pylint: disable=import-error,import-outside-toplevel

    qat = _load_lifted_qat()
    device = torch.device(args.device)

    segnet = modules.SegNet().eval().to(device)
    segnet.load_state_dict(load_file(modules.segnet_sd_path, device=str(device)))
    for parameter in segnet.parameters():
        parameter.requires_grad_(False)

    # The live trainer resolves --cache into BOTH input_cache and target_cache
    # (train_semantic_quantized_resumable.py:944-947), and both arms launched
    # with a single --cache, so conditioning and target are the same tensor here.
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    conditioning_tokens = cache["seg"].long()
    target_tokens = conditioning_tokens
    if conditioning_tokens.shape[0] != N_TOTAL_PAIRS:
        raise ValueError("semantic cache must contain all 600 pairs")
    pair_ids = torch.arange(args.pairs, dtype=torch.long)

    rows: list[dict] = []
    for label, path in arms:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("deployment_weights") != "ema_shadow":
            raise ValueError(
                f"{label}: deployment_weights is "
                f"{checkpoint.get('deployment_weights')!r}, expected 'ema_shadow' — "
                "the training arms score the EMA shadow, so anything else is a "
                "different object than the endpoint this curve is compared against"
            )
        architecture = checkpoint["architecture_config"]
        state_dict = checkpoint["state_dict"]
        sha = _sha256(path)

        model = qat.SemanticTokenRenderer(
            width=architecture["width"],
            blocks=architecture["blocks"],
            frame_dim=architecture["frame_dim"],
            num_pairs=architecture["num_pairs"],
            num_tokens=architecture["num_tokens"],
            phase_y=architecture["phase_y"],
            phase_x=architecture["phase_x"],
            temporal_radius=architecture["temporal_radius"],
        ).to(device)

        for k in ladder:
            plan = _drop_plan(state_dict, k)
            model.load_state_dict(_apply_drop(state_dict, plan))

            # REFUSE on a NaN render. MEASURED 2026-08-17: zeroing ANY FiLM row
            # makes the TRAINER-SIDE quantiser emit NaN frames. (Wording corrected
            # 2026-08-17 -- "deployed" was retracted at the memo header and must not
            # survive here: the SHIPPING receiver `_decode_row_prune` quantizes only
            # the KEPT rows and scatters them into `torch.zeros`, so it never hands a
            # zero-`amax` row to a quantiser at all. This is a trainer path.)
            # `fake_quantize` (lifted/train_semantic_quantized.py:50-54) computes
            # `scale = amax.clamp_min(1e-8) / limit` = 1.4286e-9 at bits=4 and then
            # casts it to fp16 on the NEXT line -- below fp16's smallest subnormal
            # (5.96e-8), so the cast flushes it to 0.0 and `source/scale` is 0/0.
            # The clamp_min guard is destroyed by the cast meant to follow it.
            # SCOPE, corrected: this fires at EVERY bit depth (measured 2,3,4,5,8,16)
            # because 1e-8 is below fp16 subnormal after division by any limit >= 1 --
            # bits=4 is the instance, not the boundary. And the same clamp-then-cast
            # lives in THREE implementations, not one: this trainer, plus
            # `editability_levers.deployed_fake_quant` and the packer
            # `ddm_sd1_semantic_rd_curve.quantized_tensor`. Any regression must pin
            # `amax == 0 -> finite` in all three.
            # SegNet then predicts one class everywhere and the mismatch count is a
            # CONSTANT (59,551,382 = 50.48%) identical across k AND across
            # independently-trained checkpoints -- a degenerate number wearing a
            # measurement's clothes. DEVICE CAVEAT (measured): that constant is
            # MPS-specific. On MPS SegNet silently absorbs NaN and predicts
            # Undrivable-everywhere; on CPU the NaN propagates and argmax returns
            # Road-everywhere (90,557,431 = 76.77%). So this probe must sit on the
            # RENDER, before the scorer -- a NaN guard placed after SegNet would be
            # invisible on the MPS axis entirely.
            with torch.no_grad():
                probe_ids = pair_ids[:1]
                probe = qat.render_quantized(
                    model,
                    qat.gather_conditioning(
                        conditioning_tokens, probe_ids, model.temporal_radius
                    ).to(device),
                    probe_ids.to(device),
                    args.bits,
                    exact_path=True,
                )
                if bool(torch.isnan(probe).any()):
                    raise ValueError(
                        f"{label} k={k}: render is NaN -- the ablated weights do "
                        "not survive the TRAINER-side quantiser (the shipping receiver "
                        "prunes before quantizing and never sees a zero-amax row), "
                        "so any d_seg from "
                        "them is a degenerate constant, not a measurement. A row "
                        "'dropped' by zeroing is NOT the row the receiver drops; "
                        "model the receiver's own reconstruction instead."
                    )

            d_seg = _evaluate_semantic_pairs(
                qat=qat,
                model=model,
                segnet=segnet,
                conditioning_tokens=conditioning_tokens,
                target_tokens=target_tokens,
                bits=args.bits,
                pair_ids=pair_ids,
                batch_size=args.eval_batch_size,
                device=device,
            )
            if k == 0 and label in expect_k0:
                expected = expect_k0[label]
                if d_seg != expected:
                    raise ValueError(
                        f"{label}: k=0 POSITIVE CONTROL FAILED -- measured {d_seg!r} "
                        f"vs the arm's own endpoint {expected!r} (delta "
                        f"{d_seg - expected!r}). The harness is not scoring the same "
                        "object as the arm; check --device, --eval-batch-size, "
                        "--bits, --pairs and that the checkpoint carries the EMA "
                        "shadow. Every ablation row would otherwise be off-axis."
                    )

            row = {
                "arm": label,
                "checkpoint": str(path),
                "checkpoint_sha256": sha,
                "k_rows_dropped_per_tensor": k,
                "dropped_rows": plan,
                "d_seg": d_seg,
                "flips": round(d_seg * TOTAL_PIXELS),
                "pairs": args.pairs,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "schema": "tac.film_row_ablation_curve.v1",
                "axis": "[macOS-MPS training-signal] quantized_exact_seg, EMA shadow",
                "score_claim": False,
                "ranking": "editability_levers.film_row_order (descending L2^2)",
                "k_ladder": ladder,
                "k0_positive_control": expect_k0 or None,
                "eval_batch_size": args.eval_batch_size,
                "device": args.device,
                "bits": args.bits,
                "rows": rows,
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
