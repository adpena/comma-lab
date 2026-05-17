#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Z6 predictor liveness and ego-motion smoke-condition probe.

The Z6 identity-predictor disambiguator showed identity slightly ahead on the
current synthetic smoke proxy. This probe checks whether that is a valid
predictive-coding falsification or an engineering/curriculum artifact.

It is planning evidence only: no score claim, no rank/kill authority, and no
promotion. It measures gradient flow through the FiLM predictor and whether
the smoke actually exercises the ego-motion side information.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.substrates.time_traveler_l5_z6 import (  # noqa: E402
    Z6PredictiveCodingConfig,
    Z6PredictiveCodingSubstrate,
)

SCHEMA = "z6_predictor_liveness_probe_v1"
PROBE_ID = "z6_predictor_liveness_and_ego_motion_cargo_cult_probe"
LANE_ID = "lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516"
DEFAULT_OUTPUT_JSON = (
    ".omx/research/l5_v2_z6_predictor_liveness_probe_20260516_codex.json"
)
DEFAULT_OUTPUT_MD = (
    ".omx/research/l5_v2_z6_predictor_liveness_probe_20260516_codex.md"
)
FALSE_AUTHORITY_FLAGS = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "ready_for_paid_dispatch": False,
    "paradigm_claim_allowed": False,
}
EgoMode = Literal["zero", "ramp", "random"]


def _grad_l2(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def _tiny_cfg(*, identity_predictor: bool) -> Z6PredictiveCodingConfig:
    return Z6PredictiveCodingConfig(
        latent_dim=8,
        decoder_embed_dim=16,
        decoder_channels=(12, 10, 8, 6),
        decoder_num_upsample_blocks=4,
        num_pairs=5,
        output_height=48,
        output_width=64,
        predictor_hidden_dim=16,
        predictor_film_mlp_hidden_dim=8,
        predictor_ego_motion_dim=4,
        identity_predictor=identity_predictor,
        lambda_residual_entropy=1.0,
    )


def _set_ego_motion(
    substrate: Z6PredictiveCodingSubstrate,
    *,
    mode: EgoMode,
    generator: torch.Generator,
) -> None:
    if mode == "zero":
        substrate.ego_motion_buffer.zero_()
        return
    if mode == "ramp":
        values = torch.linspace(
            -1.0,
            1.0,
            steps=substrate.cfg.num_pairs * substrate.cfg.predictor_ego_motion_dim,
        ).view(substrate.cfg.num_pairs, substrate.cfg.predictor_ego_motion_dim)
        substrate.ego_motion_buffer.copy_(values)
        return
    if mode == "random":
        substrate.ego_motion_buffer.copy_(
            torch.randn(
                substrate.cfg.num_pairs,
                substrate.cfg.predictor_ego_motion_dim,
                generator=generator,
            )
        )
        return
    raise ValueError(f"unknown ego_motion mode: {mode!r}")


def _smooth_targets(cfg: Z6PredictiveCodingConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic temporally smooth synthetic RGB targets."""

    pair_axis = torch.linspace(0.0, 1.0, steps=cfg.num_pairs).view(
        cfg.num_pairs, 1, 1, 1
    )
    y = torch.linspace(0.0, 1.0, steps=cfg.output_height).view(1, 1, cfg.output_height, 1)
    x = torch.linspace(0.0, 1.0, steps=cfg.output_width).view(1, 1, 1, cfg.output_width)
    base = 0.35 + 0.25 * torch.sin(2.0 * math.pi * (x + pair_axis))
    channel = torch.tensor([0.90, 1.00, 1.10]).view(1, 3, 1, 1)
    frame0 = (base * channel).clamp(0.0, 1.0)
    frame1 = (base.roll(shifts=1, dims=-1) * channel + 0.03 * y).clamp(0.0, 1.0)
    return frame0, frame1


def _output_delta_vs_zero_ego(
    substrate: Z6PredictiveCodingSubstrate,
) -> float | None:
    if substrate.cfg.identity_predictor:
        return None
    with torch.no_grad():
        z_prev = substrate.latent_init.unsqueeze(0)
        zero_ego = torch.zeros(1, substrate.cfg.predictor_ego_motion_dim)
        observed_ego = substrate.ego_motion_buffer[1].unsqueeze(0)
        zero_pred = substrate.predictor(z_prev, zero_ego)
        observed_pred = substrate.predictor(z_prev, observed_ego)
        return float((observed_pred - zero_pred).pow(2).sum().sqrt().item())


def measure_liveness_row(
    *,
    ego_motion_mode: EgoMode,
    identity_predictor: bool,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure one Z6 liveness row under a deterministic tiny setup."""

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 17)
    cfg = _tiny_cfg(identity_predictor=identity_predictor)
    substrate = Z6PredictiveCodingSubstrate(cfg)
    _set_ego_motion(substrate, mode=ego_motion_mode, generator=generator)
    target0, target1 = _smooth_targets(cfg)
    idx = torch.arange(cfg.num_pairs, dtype=torch.long)

    substrate.zero_grad(set_to_none=True)
    rgb0, rgb1, _z = substrate.reconstruct_pair(idx)
    recon_loss = (rgb0 - target0).pow(2).mean() + (rgb1 - target1).pow(2).mean()
    residual_loss = cfg.lambda_residual_entropy * substrate.residuals.pow(2).mean()
    loss = recon_loss + residual_loss
    loss.backward()

    predictor_params = list(substrate.predictor.parameters())
    predictor_grad_l2 = _grad_l2(predictor_params)
    residual_grad_l2 = _grad_l2([substrate.residuals])
    latent_init_grad_l2 = _grad_l2([substrate.latent_init])
    ego_nonzero_fraction = float(
        (substrate.ego_motion_buffer.detach().abs() > 0).float().mean().item()
    )
    output_delta = _output_delta_vs_zero_ego(substrate)
    predictor_param_count = sum(param.numel() for param in predictor_params)
    predictor_gradient_live = predictor_grad_l2 > 0.0
    film_conditioning_exercised = (
        (not identity_predictor)
        and ego_nonzero_fraction > 0.0
        and output_delta is not None
        and output_delta > 0.0
    )
    if identity_predictor:
        verdict = "identity_control_no_predictor_params"
    elif not film_conditioning_exercised:
        verdict = "predictor_gradient_live_but_film_conditioning_unexercised"
    elif predictor_gradient_live:
        verdict = "predictor_and_film_conditioning_live_proxy_only"
    else:
        verdict = "predictor_gradient_missing"

    return {
        "mode": (
            "identity_predictor_control"
            if identity_predictor
            else f"full_film_{ego_motion_mode}_ego"
        ),
        "ego_motion_mode": ego_motion_mode,
        "identity_predictor": identity_predictor,
        "loss_proxy": float(loss.detach().item()),
        "recon_loss_proxy": float(recon_loss.detach().item()),
        "residual_loss_proxy": float(residual_loss.detach().item()),
        "predictor_param_count": predictor_param_count,
        "predictor_gradient_l2": predictor_grad_l2,
        "residual_gradient_l2": residual_grad_l2,
        "latent_init_gradient_l2": latent_init_grad_l2,
        "ego_motion_nonzero_fraction": ego_nonzero_fraction,
        "ego_motion_requires_grad": bool(substrate.ego_motion_buffer.requires_grad),
        "predictor_output_delta_vs_zero_ego_l2": output_delta,
        "predictor_gradient_live": predictor_gradient_live,
        "film_conditioning_exercised": film_conditioning_exercised,
        "verdict": verdict,
    }


def build_probe_payload(*, seed: int = 0) -> dict[str, Any]:
    """Build the full fail-closed liveness payload."""

    rows = [
        measure_liveness_row(
            ego_motion_mode="zero",
            identity_predictor=False,
            seed=seed,
        ),
        measure_liveness_row(
            ego_motion_mode="ramp",
            identity_predictor=False,
            seed=seed,
        ),
        measure_liveness_row(
            ego_motion_mode="zero",
            identity_predictor=True,
            seed=seed,
        ),
    ]
    zero_full = rows[0]
    ramp_full = rows[1]
    if (
        zero_full["predictor_gradient_live"]
        and not zero_full["film_conditioning_exercised"]
        and ramp_full["film_conditioning_exercised"]
    ):
        verdict = "z6_predictor_live_smoke_film_conditioning_cargo_cult"
    elif not zero_full["predictor_gradient_live"]:
        verdict = "z6_predictor_gradient_missing_engineering_bug"
    else:
        verdict = "z6_predictor_liveness_indeterminate"
    return {
        "schema": SCHEMA,
        "probe_id": PROBE_ID,
        "lane_id": LANE_ID,
        "seed": seed,
        "evidence_grade": "synthetic_liveness_probe_no_scorer",
        "verdict": verdict,
        **FALSE_AUTHORITY_FLAGS,
        "rows": rows,
        "findings": [
            "full FiLM predictor receives nonzero gradients in the smoke graph",
            "current zero-ego smoke does not exercise FiLM conditioning",
            "ramp ego-motion control changes predictor output and confirms the conditioning path is live",
            "identity-vs-full smoke result is not a fair falsification of ego-motion predictive coding",
        ],
        "blockers": [
            "current_z6_smoke_uses_zero_ego_motion_buffer",
            "current_z6_smoke_uses_synthetic_targets_without_real_temporal_pose_signal",
            "no_scorer_load",
            "no_contest_cpu_cuda_pair",
            "not_score_or_paradigm_authority",
        ],
        "recommended_next_actions": [
            "populate ego_motion_buffer from real-video PoseNet/proxy features before rerunning identity disambiguator",
            "rerun Z6 identity disambiguator on real-video paired smoke after ego-motion population",
            "keep full_main council-gated until predictor-vs-identity wins on a real signal axis",
        ],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# L5 v2 Z6 predictor liveness probe",
        "",
        f"- schema: `{payload.get('schema')}`",
        f"- probe_id: `{payload.get('probe_id')}`",
        f"- lane_id: `{payload.get('lane_id')}`",
        f"- evidence_grade: `{payload.get('evidence_grade')}`",
        f"- verdict: `{payload.get('verdict')}`",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        "- rank_or_kill_eligible: `false`",
        "- ready_for_exact_eval_dispatch: `false`",
        "- ready_for_paid_dispatch: `false`",
        "- paradigm_claim_allowed: `false`",
        "",
        "This is an engineering liveness/cargo-cult probe. It explains whether "
        "the Z6 identity-vs-full smoke result exercised the predictor path; it "
        "does not claim score movement.",
        "",
        "## Rows",
    ]
    rows = payload.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            lines.extend(
                [
                    "",
                    f"### {row.get('mode')}",
                    "",
                    f"- verdict: `{row.get('verdict')}`",
                    f"- predictor_param_count: `{row.get('predictor_param_count')}`",
                    f"- predictor_gradient_l2: `{row.get('predictor_gradient_l2')}`",
                    f"- ego_motion_nonzero_fraction: `{row.get('ego_motion_nonzero_fraction')}`",
                    f"- predictor_output_delta_vs_zero_ego_l2: `{row.get('predictor_output_delta_vs_zero_ego_l2')}`",
                    f"- film_conditioning_exercised: `{row.get('film_conditioning_exercised')}`",
                ]
            )
    for section, key in (
        ("Findings", "findings"),
        ("Blockers", "blockers"),
        ("Recommended Next Actions", "recommended_next_actions"),
    ):
        values = payload.get(key)
        if isinstance(values, list):
            lines.extend(["", f"## {section}"])
            for value in values:
                lines.append(f"- {value}")
    return "\n".join(lines) + "\n"


def _resolve_output(path: Path, *, repo_root: Path) -> Path:
    resolved = path if path.is_absolute() else repo_root / path
    resolved = resolved.expanduser().resolve()
    resolved.relative_to(repo_root)
    text = str(resolved)
    if (
        text.startswith("/tmp/")
        or "/private/tmp/" in text
        or "/var/tmp/" in text
    ):
        raise ValueError(f"refusing to write Z6 liveness output to tmp: {text!r}")
    return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=Path(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", type=Path, default=Path(DEFAULT_OUTPUT_MD))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    try:
        payload = build_probe_payload(seed=args.seed)
        output_json = _resolve_output(args.output_json, repo_root=repo_root)
        output_md = _resolve_output(args.output_md, repo_root=repo_root)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(payload), encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"[z6-liveness] FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        "[z6-liveness] "
        f"verdict={payload['verdict']} "
        f"evidence_grade={payload['evidence_grade']} "
        "score_claim=false promotion_eligible=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
