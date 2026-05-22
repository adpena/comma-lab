# SPDX-License-Identifier: MIT
"""CUDA T4 eval pipeline for MLX-trained weights.

OVERNIGHT-WW Phase 4 per operator directive 2026-05-21. This module is the
canonical glue between MLX-native training and CUDA T4 contest-axis
evaluation. It does NOT dispatch to Modal — that remains the operator's
explicit responsibility per CLAUDE.md "Executing actions with care" — but
it documents the canonical invocation pattern + builds the recipe stub the
operator can hand to ``tools/operator_authorize.py``.

Canonical pipeline:

    1. MLX-native training (FREE local on M5 Max)
       e.g. tac.substrates.grayscale_lut.mlx_native.GrayscaleLutMLXNative
       trained against contest video via :mod:`tac.portable_primitives` +
       :mod:`tac.portable_primitives.optim` + :mod:`tac.portable_primitives.loss`.

    2. Export (this module's sister :mod:`mlx_to_pytorch_export`)
       state_dict (numpy) -> .pt file (PyTorch canonical layout)

    3. Pack archive (canonical substrate-specific archive grammar)
       e.g. tac.substrates.grayscale_lut.archive.pack_archive

    4. Operator-routable CUDA T4 dispatch
       :func:`build_cuda_t4_eval_invocation` returns the canonical command
       string + recipe filename + expected paid cost

    5. Modal call_id ledger registration per Catalog #245
       (automatic via :mod:`tac.deploy.modal.call_id_ledger`)

    6. Contest-axis [contest-CUDA] score
       canonical Provenance per Catalog #287/#323

Per CLAUDE.md non-negotiables PRESERVED:
- This module NEVER calls Modal directly (no spawn / no dispatch).
- The operator MUST explicitly run the returned command after reviewing
  the archive + .pt file + recipe per Catalog #199 paired-env bypass
  discipline.
- The non-promotable evidence_grade is preserved through the pipeline;
  only the final CUDA T4 eval (Step 6) produces a promotable
  [contest-CUDA] anchor per CLAUDE.md "MPS auth eval is NOISE" +
  Catalog #192.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "build_cuda_t4_eval_invocation",
    "describe_pipeline_steps",
]


def build_cuda_t4_eval_invocation(
    *,
    substrate_id: str,
    archive_path: Path | str,
    archive_sha256: str,
    expected_cost_usd: float = 0.30,
    target: str = "modal",
) -> dict[str, Any]:
    """Return the canonical CUDA T4 eval invocation for an MLX-trained archive.

    Per CLAUDE.md "Executing actions with care": this module does NOT
    invoke the dispatch — it returns the canonical command string + recipe
    parameters the operator reviews + runs explicitly.

    Args:
        substrate_id: canonical substrate identifier (e.g. "grayscale_lut").
        archive_path: path to the packed archive.zip (per substrate-specific
            archive grammar; e.g. GLV1 for Selfcomp grayscale_lut).
        archive_sha256: sha256 of the archive bytes; required for canonical
            custody per Catalog #127.
        expected_cost_usd: estimated paid GPU cost. T4 smoke ~$0.10-0.30;
            full eval ~$0.30-0.60. Used for cost-band classification per
            :mod:`tac.cost_band_calibration` and Catalog #199 paired-env
            bypass threshold.
        target: dispatch target. Supported: "modal" (canonical CUDA T4),
            "local-cpu" (free GHA-equivalent Linux x86_64 advisory).

    Returns:
        Dict with:
        - ``operator_command``: shell command for operator to invoke
        - ``recipe_name``: canonical recipe basename
        - ``recipe_yaml_path``: expected path to recipe under
          ``.omx/operator_authorize_recipes/``
        - ``cost_band``: classification per cost ($0.30 -> "smoke")
        - ``expected_cost_usd``: passed through
        - ``next_steps``: ordered list of operator-routable steps
    """
    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"archive path does not exist: {archive}")

    if len(archive_sha256) != 64:
        raise ValueError(
            f"archive_sha256 must be 64 hex chars; got {len(archive_sha256)}"
        )
    try:
        int(archive_sha256, 16)
    except ValueError as exc:
        raise ValueError(f"archive_sha256 must be hex: {archive_sha256!r}") from exc

    if target not in {"modal", "local-cpu"}:
        raise ValueError(f"unsupported target {target!r}; use 'modal' or 'local-cpu'")

    recipe_name = f"substrate_{substrate_id}_mlx_trained_eval_{target}_t4_dispatch"
    recipe_yaml_path = f".omx/operator_authorize_recipes/{recipe_name}.yaml"

    # Classify cost-band per Catalog #199 paired-env bypass threshold.
    if expected_cost_usd < 0.50:
        cost_band = "smoke"
    elif expected_cost_usd < 5.0:
        cost_band = "full"
    else:
        cost_band = "long_burn"

    operator_command = (
        f"tools/operator_authorize.py --recipe {recipe_name} --target {target}"
    )
    if expected_cost_usd >= 1.0:
        operator_command += (
            f" # paid {target}; estimated ${expected_cost_usd:.2f} on CUDA T4"
        )

    next_steps = [
        f"1. Stage archive.zip at {archive.resolve()} (sha256={archive_sha256[:12]}...)",
        f"2. Author / verify recipe at {recipe_yaml_path} (set "
        f"`dispatch_enabled: false` until operator-approved review)",
        f"3. Run pre-flight: tools/local_pre_deploy_check.py --strict --recipe {recipe_name}",
        f"4. Set OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 + "
        f"OPERATOR_AUTHORIZE_SESSION_BUDGET_USD={max(expected_cost_usd * 2, 1.0):.2f} per Catalog #199",
        f"5. Operator-explicitly invoke: {operator_command}",
        "6. Harvest via tools/parallel_harvest_actuator.py --call-id <fc-...> "
        "(canonical Modal call_id ledger per Catalog #245)",
        "7. Verify [contest-CUDA] score lands in canonical Provenance per Catalog #287/#323",
    ]

    return {
        "operator_command": operator_command,
        "recipe_name": recipe_name,
        "recipe_yaml_path": recipe_yaml_path,
        "cost_band": cost_band,
        "expected_cost_usd": expected_cost_usd,
        "archive_path": str(archive.resolve()),
        "archive_sha256": archive_sha256,
        "substrate_id": substrate_id,
        "target": target,
        "next_steps": next_steps,
        # Canonical Provenance markers carried until the final eval lands.
        "current_evidence_grade": "macOS-MLX-research-signal",
        "promotion_blockers": [
            "requires_cuda_t4_or_linux_x86_64_paired_eval",
            "operator_authorize_chain_not_yet_invoked",
            "modal_call_id_not_yet_registered",
        ],
        "promotion_unlock_event": (
            "[contest-CUDA] score lands via canonical Modal call_id ledger "
            "after operator-explicit dispatch"
        ),
    }


def describe_pipeline_steps() -> list[dict[str, str]]:
    """Return human-readable description of the 6-step canonical pipeline.

    Used by operator-facing documentation + the OVERNIGHT-WW landing memo
    to make the train-anywhere-eval-anywhere pattern explicit.
    """
    return [
        {
            "step": "1",
            "name": "MLX-native training (FREE local on M5 Max)",
            "module": "tac.substrates.<sub>.mlx_native + tac.portable_primitives",
            "cost": "$0",
            "axis": "macOS-MLX-research-signal (non-promotable per Catalog #1+#192)",
        },
        {
            "step": "2",
            "name": "Export weights MLX -> PyTorch .pt",
            "module": "tac.local_acceleration.mlx_to_pytorch_export",
            "cost": "$0 (local CPU serialization)",
            "axis": "preserves training axis (non-promotable until Step 4 lands)",
        },
        {
            "step": "3",
            "name": "Pack archive.zip (substrate-specific grammar)",
            "module": "tac.substrates.<sub>.archive.pack_archive",
            "cost": "$0 (local CPU)",
            "axis": "preserves training axis",
        },
        {
            "step": "4",
            "name": "Operator-routable CUDA T4 dispatch",
            "module": "tools/operator_authorize.py --target modal",
            "cost": "$0.10-0.60 (T4 smoke or full)",
            "axis": "paid Linux x86_64 + NVIDIA T4 (1:1 contest-compliant)",
        },
        {
            "step": "5",
            "name": "Modal call_id ledger registration",
            "module": "tac.deploy.modal.call_id_ledger (Catalog #245)",
            "cost": "automatic (no extra spend)",
            "axis": "canonical Provenance per Catalog #287/#323",
        },
        {
            "step": "6",
            "name": "Contest-axis [contest-CUDA] score",
            "module": "experiments/contest_auth_eval.py via canonical helper",
            "cost": "included in Step 4",
            "axis": "[contest-CUDA] promotion-eligible (per CLAUDE.md "
            "'Submission auth eval — BOTH CPU AND CUDA')",
        },
    ]
