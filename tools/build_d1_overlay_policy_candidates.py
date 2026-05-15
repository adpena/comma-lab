#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize byte-closed D1 overlay policy candidates.

This tool does not score or promote candidates. It rewrites only the D1POLY1
metadata fields that select inflate-time overlay policy, then emits a complete
submission runtime tree and deterministic archive.zip for each policy tuple.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.train_substrate_d1_segnet_margin_polytope import (  # noqa: E402
    _build_archive_zip,
    _write_runtime,
)
from tac.substrates.d1_segnet_margin_polytope import (  # noqa: E402
    D1PolytopeConfig,
    analyze_d1_overlay_effect,
    encode_polytope_payload,
    pack_archive,
    parse_archive,
    update_d1poly1_meta,
)
from tac.substrates.d1_segnet_margin_polytope.overlay import (  # noqa: E402
    D1_OVERLAY_AMPLITUDE_SCALES,
    D1_OVERLAY_CHANNEL_POLICIES,
    D1_OVERLAY_SIGN_POLICIES,
    channel_policy_weights,
    normalize_overlay_amplitude_scale,
    overlay_sign_for_pair,
    pack_pair_sign_mask,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _parse_policies(raw: str) -> list[str]:
    policies = [item.strip() for item in raw.split(",") if item.strip()]
    if not policies:
        raise SystemExit("--policies produced an empty policy list")
    allowed = set(D1_OVERLAY_CHANNEL_POLICIES)
    unknown = [policy for policy in policies if policy not in allowed]
    if unknown:
        raise SystemExit(
            f"unsupported policies {unknown}; expected subset of {sorted(allowed)}"
        )
    return policies


def _parse_sign_policies(raw: str) -> list[str]:
    policies = [item.strip() for item in raw.split(",") if item.strip()]
    if not policies:
        raise SystemExit("--sign-policies produced an empty policy list")
    allowed = set(D1_OVERLAY_SIGN_POLICIES)
    unknown = [policy for policy in policies if policy not in allowed]
    if unknown:
        raise SystemExit(
            f"unsupported sign policies {unknown}; expected subset of {sorted(allowed)}"
        )
    return policies


def _parse_amplitude_scales(raw: str) -> list[float]:
    scales: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            scales.append(normalize_overlay_amplitude_scale(float(item)))
        except ValueError as exc:
            raise SystemExit(f"unsupported amplitude scale {item!r}: {exc}") from exc
    if not scales:
        raise SystemExit("--amplitude-scales produced an empty scale list")
    return scales


def _parse_budget_bits(raw: str) -> list[int]:
    budgets: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        budget = int(item)
        if budget <= 0:
            raise SystemExit(f"budget bits must be > 0; got {budget}")
        budgets.append(budget)
    return budgets


def _parse_lipschitz_values(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0.0:
            raise SystemExit(f"jacobian_lipschitz must be > 0; got {value}")
        values.append(value)
    return values


def _parse_resolution(raw: str) -> tuple[int, int] | None:
    text = raw.strip().lower()
    if not text:
        return None
    if "x" not in text:
        raise SystemExit("--margin-map-resolution must be formatted as HxW")
    h_text, w_text = text.split("x", 1)
    h = int(h_text)
    w = int(w_text)
    if h <= 0 or w <= 0:
        raise SystemExit(f"margin-map resolution must be positive; got {h}x{w}")
    return h, w


def _load_pair_sign_mask(path: Path | None) -> tuple[int, ...] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_signs = payload
    elif isinstance(payload, dict):
        raw_signs = payload.get("pair_signs")
        if raw_signs is None:
            selector = payload.get("selector", {})
            raw_signs = selector.get("pair_signs") if isinstance(selector, dict) else None
    else:
        raw_signs = None
    if not isinstance(raw_signs, list) or not raw_signs:
        raise SystemExit(
            "--pair-sign-mask-json must contain a non-empty pair_signs list"
        )
    signs = tuple(int(value) for value in raw_signs)
    bad = [value for value in signs if value not in (-1, 0, 1)]
    if bad:
        raise SystemExit(
            f"--pair-sign-mask-json values must be -1, 0, or 1; got {bad[:8]}"
        )
    return signs


def _scale_slug(scale: float) -> str:
    text = f"{normalize_overlay_amplitude_scale(scale):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _value_slug(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def _candidate_id(
    *,
    channel_policy: str,
    amplitude_scale: float,
    sign_policy: str,
    payload_budget_bits: int | None,
    jacobian_lipschitz: float | None,
    margin_map_resolution: tuple[int, int] | None,
    pair_mask_label: str | None,
) -> str:
    prefix = "d1_overlay"
    if payload_budget_bits is not None and jacobian_lipschitz is not None:
        prefix += (
            f"_budget_{int(payload_budget_bits)}"
            f"_L_{_value_slug(float(jacobian_lipschitz))}"
        )
    if margin_map_resolution is not None:
        prefix += f"_res_{margin_map_resolution[0]}x{margin_map_resolution[1]}"
    return (
        f"{prefix}_channel_{channel_policy}"
        f"_amp_{_scale_slug(amplitude_scale)}"
        f"_sign_{sign_policy}"
        + (f"_pairmask_{pair_mask_label}" if sign_policy == "pair_mask" else "")
    )


def _overlay_effect_equivalence_key(
    *,
    channel_policy: str,
    amplitude_scale: float,
    sign_policy: str,
    payload_budget_bits: int | None,
    jacobian_lipschitz: float | None,
    margin_map_resolution: tuple[int, int] | None,
    pair_sign_mask_sha256: str | None,
    pair_sign_mask: tuple[int, ...] | None,
) -> str:
    """Return a key for variants that produce the same signed frame deltas.

    ``green + negate_payload`` and ``neg_green + payload`` differ in metadata
    and ZIP bytes, but they add the same pixel deltas to every pair. The key
    lets the manifest mark those duplicates before another paired auth eval
    spends time on an effect-equivalent packet.
    """
    weights = channel_policy_weights(channel_policy).astype(int)
    pair0 = (
        weights
        * overlay_sign_for_pair(
            sign_policy,
            0,
            pair_sign_mask if sign_policy == "pair_mask" else None,
        )
    ).tolist()
    pair1 = (
        weights
        * overlay_sign_for_pair(
            sign_policy,
            1,
            pair_sign_mask if sign_policy == "pair_mask" else None,
        )
    ).tolist()
    resolution = (
        f"{margin_map_resolution[0]}x{margin_map_resolution[1]}"
        if margin_map_resolution is not None
        else "source"
    )
    return "|".join(
        [
            f"budget={payload_budget_bits}",
            f"L={jacobian_lipschitz}",
            f"res={resolution}",
            f"amp={normalize_overlay_amplitude_scale(amplitude_scale):.6g}",
            f"pair0={pair0}",
            f"pair1={pair1}",
            f"pair_mask_sha256={pair_sign_mask_sha256 or ''}",
        ]
    )


def _rebuilt_payload_source(
    *,
    source,
    source_d1_sha256: str,
    base_sha256: str,
    base_bytes: int,
    payload_budget_bits: int,
    jacobian_lipschitz: float,
    margin_map_resolution: tuple[int, int] | None,
) -> bytes:
    import torch
    import torch.nn.functional as F

    margin_map = torch.from_numpy(source.margin_map_float())
    target_resolution = margin_map_resolution or (source.height, source.width)
    if tuple(margin_map.shape) != tuple(target_resolution):
        margin_map = (
            F.interpolate(
                margin_map.unsqueeze(0).unsqueeze(0).float(),
                size=target_resolution,
                mode="area",
            )
            .squeeze(0)
            .squeeze(0)
            .clamp_min(0.0)
            .contiguous()
        )
    cfg = D1PolytopeConfig(
        base_substrate_id=source.base_substrate_id,
        margin_map_mode=str(source.meta.get("margin_map_mode", "segnet_top1_minus_top2")),
        polytope_payload_bits=int(payload_budget_bits),
        margin_map_resolution=target_resolution,
        margin_map_int8_scale=float(source.meta.get("margin_map_int8_scale", 127.0)),
        jacobian_lipschitz=float(jacobian_lipschitz),
        margin_threshold=float(source.meta.get("margin_threshold", 0.1)),
        pose_sqrt_weight=float(source.meta.get("pose_sqrt_weight", 10.0**0.5)),
        seg_weight=float(source.meta.get("seg_weight", 100.0)),
    )
    polytope_payload = encode_polytope_payload(
        margin_map,
        jacobian_lipschitz=float(jacobian_lipschitz),
        budget_bits=int(payload_budget_bits),
    )
    return pack_archive(
        margin_map=margin_map,
        polytope_payload=polytope_payload,
        jacobian_lipschitz=float(jacobian_lipschitz),
        base_substrate_id=source.base_substrate_id,
        base_archive_sha256=base_sha256,
        base_archive_bytes=base_bytes,
        config=cfg,
        extra_meta={
            "payload_sweep_candidate": True,
            "source_d1_bin_sha256": source_d1_sha256,
            "source_polytope_payload_bits": int(
                source.meta.get("polytope_payload_bits", 0)
            ),
            "source_jacobian_lipschitz": float(source.jacobian_lipschitz),
            "source_margin_map_resolution": [source.height, source.width],
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build byte-closed D1 overlay policy candidates"
    )
    parser.add_argument("--d1-bin", type=Path, required=True)
    parser.add_argument("--a1-bin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--policies",
        default=",".join(D1_OVERLAY_CHANNEL_POLICIES),
        help="Comma-separated channel policies to materialize.",
    )
    parser.add_argument(
        "--amplitude-scales",
        default="1.0",
        help=(
            "Comma-separated overlay attenuation scales in [0,1]. "
            f"Known schedule: {','.join(str(v) for v in D1_OVERLAY_AMPLITUDE_SCALES)}."
        ),
    )
    parser.add_argument(
        "--sign-policies",
        default="payload",
        help="Comma-separated sign policies to materialize.",
    )
    parser.add_argument(
        "--payload-budget-bits",
        default="",
        help=(
            "Optional comma-separated payload budgets. When set, rebuild "
            "the D1 polytope payload from the source margin map instead of "
            "only mutating policy metadata."
        ),
    )
    parser.add_argument(
        "--jacobian-lipschitz",
        default="",
        help=(
            "Optional comma-separated L values for payload rebuild sweeps. "
            "Defaults to the source D1 archive value when --payload-budget-bits "
            "is set."
        ),
    )
    parser.add_argument(
        "--margin-map-resolution",
        default="",
        help=(
            "Optional payload-rebuild resolution formatted as HxW, for example "
            "96x128. Uses area downsample from the source margin map."
        ),
    )
    parser.add_argument(
        "--pair-sign-mask-json",
        type=Path,
        help=(
            "Optional JSON containing pair_signs=[-1,0,1,...]. Required when "
            "--sign-policies includes pair_mask."
        ),
    )
    parser.add_argument(
        "--pair-sign-mask-label",
        default="",
        help="Short label appended to pair_mask candidate IDs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    d1_bytes = args.d1_bin.read_bytes()
    a1_bytes = args.a1_bin.read_bytes()
    source = parse_archive(d1_bytes)
    a1_sha = _sha256_bytes(a1_bytes)
    if a1_sha[:16] != source.base_archive_sha256_truncated:
        raise SystemExit(
            "A1 base sha mismatch: "
            f"{a1_sha[:16]} != {source.base_archive_sha256_truncated}"
        )

    policies = _parse_policies(args.policies)
    amplitude_scales = _parse_amplitude_scales(args.amplitude_scales)
    sign_policies = _parse_sign_policies(args.sign_policies)
    payload_budgets = _parse_budget_bits(args.payload_budget_bits)
    lipschitz_values = _parse_lipschitz_values(args.jacobian_lipschitz)
    margin_map_resolution = _parse_resolution(args.margin_map_resolution)
    pair_sign_mask = _load_pair_sign_mask(args.pair_sign_mask_json)
    pair_sign_mask_b64: str | None = None
    pair_sign_mask_sha256: str | None = None
    pair_mask_label: str | None = None
    if pair_sign_mask is not None:
        pair_sign_mask_b64 = pack_pair_sign_mask(pair_sign_mask)
        pair_sign_mask_sha256 = hashlib.sha256(
            base64.b64decode(pair_sign_mask_b64.encode("ascii"), validate=True)
        ).hexdigest()
        pair_mask_label = args.pair_sign_mask_label.strip() or pair_sign_mask_sha256[:12]
    if "pair_mask" in sign_policies and pair_sign_mask is None:
        raise SystemExit("--sign-policies=pair_mask requires --pair-sign-mask-json")
    if pair_sign_mask is not None and "pair_mask" not in sign_policies:
        raise SystemExit(
            "--pair-sign-mask-json was provided but --sign-policies does not include pair_mask"
        )
    if margin_map_resolution and not payload_budgets:
        raise SystemExit(
            "--margin-map-resolution requires --payload-budget-bits so the "
            "payload can be rebuilt"
        )
    if payload_budgets and not lipschitz_values:
        lipschitz_values = [float(source.jacobian_lipschitz)]
    if lipschitz_values and not payload_budgets:
        raise SystemExit(
            "--jacobian-lipschitz requires --payload-budget-bits so the payload "
            "can be rebuilt"
        )
    payload_sources: list[tuple[int | None, float | None, bytes]] = [(None, None, d1_bytes)]
    if payload_budgets:
        payload_sources = []
        for budget_bits in payload_budgets:
            for lipschitz in lipschitz_values:
                payload_sources.append(
                    (
                        int(budget_bits),
                        float(lipschitz),
                        _rebuilt_payload_source(
                            source=source,
                            source_d1_sha256=_sha256_bytes(d1_bytes),
                            base_sha256=a1_sha,
                            base_bytes=len(a1_bytes),
                            payload_budget_bits=int(budget_bits),
                            jacobian_lipschitz=float(lipschitz),
                            margin_map_resolution=margin_map_resolution,
                        ),
                    )
                )
    rows: list[dict[str, Any]] = []
    for payload_budget_bits, jacobian_lipschitz, payload_source in payload_sources:
        for policy in policies:
            for amplitude_scale in amplitude_scales:
                for sign_policy in sign_policies:
                    candidate_id = _candidate_id(
                        channel_policy=policy,
                        amplitude_scale=amplitude_scale,
                        sign_policy=sign_policy,
                        payload_budget_bits=payload_budget_bits,
                        jacobian_lipschitz=jacobian_lipschitz,
                        margin_map_resolution=margin_map_resolution,
                        pair_mask_label=pair_mask_label,
                    )
                    candidate_root = args.output_dir / candidate_id
                    submission_dir = candidate_root / "submission_dir"
                    candidate_root.mkdir(parents=True, exist_ok=True)

                    d1_variant = update_d1poly1_meta(
                        payload_source,
                        {
                            "overlay_channel_policy": policy,
                            "overlay_amplitude_scale": amplitude_scale,
                            "overlay_sign_policy": sign_policy,
                            **(
                                {
                                    "overlay_pair_sign_mask_b64": pair_sign_mask_b64,
                                    "overlay_pair_sign_mask_n_pairs": len(
                                        pair_sign_mask or ()
                                    ),
                                    "overlay_pair_sign_mask_sha256": pair_sign_mask_sha256,
                                }
                                if sign_policy == "pair_mask"
                                else {}
                            ),
                        },
                    )
                    parsed_variant = parse_archive(d1_variant)
                    overlay_diag = analyze_d1_overlay_effect(
                        parsed_variant,
                        channel_policy=policy,
                        amplitude_scale=amplitude_scale,
                        sign_policy=sign_policy,
                        pair_sign_mask=(
                            pair_sign_mask
                            if sign_policy == "pair_mask"
                            else None
                        ),
                    )
                    _write_runtime(submission_dir)
                    (submission_dir / "d1_polytope.bin").write_bytes(d1_variant)
                    (submission_dir / "a1.bin").write_bytes(a1_bytes)
                    archive_zip = candidate_root / "archive.zip"
                    _build_archive_zip(
                        archive_zip,
                        d1_bin_bytes=d1_variant,
                        base_bin_bytes=a1_bytes,
                        base_substrate_id="a1",
                    )
                    archive_bytes = archive_zip.read_bytes()
                    (submission_dir / "archive.zip").write_bytes(archive_bytes)
                    row = {
                        "candidate_id": candidate_id,
                        "overlay_channel_policy": policy,
                        "overlay_amplitude_scale": amplitude_scale,
                        "overlay_sign_policy": sign_policy,
                        "candidate_root": candidate_root,
                        "submission_dir": submission_dir,
                        "archive_zip": archive_zip,
                        "archive_bytes": len(archive_bytes),
                        "archive_sha256": _sha256_bytes(archive_bytes),
                        "d1_bin_bytes": len(d1_variant),
                        "d1_bin_sha256": _sha256_bytes(d1_variant),
                        "payload_budget_bits": payload_budget_bits,
                        "jacobian_lipschitz_override": jacobian_lipschitz,
                        "margin_map_resolution_override": (
                            list(margin_map_resolution)
                            if margin_map_resolution is not None
                            else None
                        ),
                        "d1_meta_policy_tuple": {
                            "overlay_channel_policy": parsed_variant.meta.get(
                                "overlay_channel_policy"
                            ),
                            "overlay_amplitude_scale": parsed_variant.meta.get(
                                "overlay_amplitude_scale"
                            ),
                            "overlay_sign_policy": parsed_variant.meta.get(
                                "overlay_sign_policy"
                            ),
                        },
                        "overlay_effect_equivalence_key": (
                            _overlay_effect_equivalence_key(
                                channel_policy=policy,
                                amplitude_scale=amplitude_scale,
                                sign_policy=sign_policy,
                                payload_budget_bits=payload_budget_bits,
                                jacobian_lipschitz=jacobian_lipschitz,
                                margin_map_resolution=margin_map_resolution,
                                pair_sign_mask_sha256=(
                                    pair_sign_mask_sha256
                                    if sign_policy == "pair_mask"
                                    else None
                                ),
                                pair_sign_mask=(
                                    pair_sign_mask
                                    if sign_policy == "pair_mask"
                                    else None
                                ),
                            )
                        ),
                        "duplicate_of_candidate_id": None,
                        "d1_overlay_diagnostics": overlay_diag.to_json_dict(),
                        "pair_sign_mask": (
                            {
                                "n_pairs": len(pair_sign_mask or ()),
                                "active_pairs": int(
                                    sum(1 for value in (pair_sign_mask or ()) if value != 0)
                                ),
                                "positive_pairs": int(
                                    sum(1 for value in (pair_sign_mask or ()) if value > 0)
                                ),
                                "negative_pairs": int(
                                    sum(1 for value in (pair_sign_mask or ()) if value < 0)
                                ),
                                "sha256": pair_sign_mask_sha256,
                                "label": pair_mask_label,
                            }
                            if sign_policy == "pair_mask"
                            else None
                        ),
                        "source_d1_bin_sha256": _sha256_bytes(d1_bytes),
                        "a1_bin_sha256": a1_sha,
                        "score_claim": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                        "dispatch_blockers": [
                            "not_paired_contest_cpu_cuda_exact_eval",
                            "no_dispatch_claim_for_policy_candidate",
                            *overlay_diag.dispatch_blockers,
                        ],
                    }
                    rows.append(row)

    best_by_effect: dict[str, dict[str, Any]] = {}
    for row in sorted(
        rows, key=lambda item: (int(item["archive_bytes"]), item["candidate_id"])
    ):
        key = str(row["overlay_effect_equivalence_key"])
        if key not in best_by_effect:
            best_by_effect[key] = row
            continue
        canonical = best_by_effect[key]
        row["duplicate_of_candidate_id"] = canonical["candidate_id"]
        row["dispatch_blockers"].append(
            "d1_overlay_effect_duplicate_of_"
            + str(canonical["candidate_id"])
        )

    for row in rows:
        _write_json(Path(row["candidate_root"]) / "candidate_manifest.json", row)

    summary = {
        "tool": "tools/build_d1_overlay_policy_candidates.py",
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": _git_head(),
        "source_d1_bin": args.d1_bin,
        "source_a1_bin": args.a1_bin,
        "source_d1_bin_sha256": _sha256_bytes(d1_bytes),
        "a1_bin_sha256": a1_sha,
        "channel_policies": policies,
        "amplitude_scales": amplitude_scales,
        "sign_policies": sign_policies,
        "payload_budget_bits": payload_budgets,
        "jacobian_lipschitz_values": lipschitz_values,
        "margin_map_resolution": (
            list(margin_map_resolution) if margin_map_resolution is not None else None
        ),
        "pair_sign_mask": (
            {
                "n_pairs": len(pair_sign_mask or ()),
                "active_pairs": int(
                    sum(1 for value in (pair_sign_mask or ()) if value != 0)
                ),
                "positive_pairs": int(
                    sum(1 for value in (pair_sign_mask or ()) if value > 0)
                ),
                "negative_pairs": int(
                    sum(1 for value in (pair_sign_mask or ()) if value < 0)
                ),
                "sha256": pair_sign_mask_sha256,
                "label": pair_mask_label,
            }
            if pair_sign_mask is not None
            else None
        ),
        "policy_count": len(rows),
        "candidates": rows,
        "score_claim": False,
        "promotion_eligible": False,
    }
    _write_json(args.output_dir / "d1_overlay_policy_candidates_manifest.json", summary)
    print(json.dumps(summary, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
