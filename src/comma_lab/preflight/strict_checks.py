"""Public comma-lab strict preflight catalog and compatibility wrappers.

The canonical contest/runtime preflight implementation lives in
``tac.preflight``. That is intentional: submission compliance, archive safety,
CUDA-score custody, and runtime-package guards are part of the codec package's
public contract. This module is an adapter for comma-lab reports, ARA catalogs,
hosted supplements, and research dashboards.
"""

from __future__ import annotations

import argparse
import inspect
import json
from typing import Any

from tac import preflight as _tac_preflight


check_no_mps_fallback_default = _tac_preflight.check_no_mps_fallback_default
check_42_train_inference_parity = _tac_preflight.check_pose_projection_train_inference_parity
check_dispatch_cli_shell_hazards = _tac_preflight.check_dispatch_cli_shell_hazards
check_reverse_engineering_tree_curation = _tac_preflight.check_reverse_engineering_tree_curation
check_feature_flags_have_live_objective_effect = _tac_preflight.check_feature_flags_have_live_objective_effect
check_public_release_hygiene = _tac_preflight.check_public_release_hygiene
preflight_all = _tac_preflight.preflight_all


ANCHOR_CHECKS = {
    "check_no_mps_fallback_default": "device safety: no CPU/MPS fallback for authoritative score paths",
    "check_42_train_inference_parity": "train/inference parity for pose projection and codec contracts",
    "check_dispatch_cli_shell_hazards": "dispatch shell/CLI typo and local-platform hazard guard",
    "check_reverse_engineering_tree_curation": "reverse-engineering tree custody and raw-artifact guard",
    "check_feature_flags_have_live_objective_effect": "dead flag and no-op objective guard",
    "check_public_release_hygiene": "public docs/site/notebook private-surface guard",
}


def emit_catalog() -> dict[str, Any]:
    """Return a machine-readable strict-check catalog for paper/ARA tooling."""
    check_names = sorted(
        name
        for name, value in inspect.getmembers(_tac_preflight)
        if name.startswith("check_") and callable(value)
    )
    return {
        "title": "comma-lab strict preflight check catalog",
        "source_of_truth": "src/tac/preflight.py",
        "adapter": "src/comma_lab/preflight/strict_checks.py",
        "migration_note": (
            "Canonical preflight lives in tac.preflight. comma_lab exposes a "
            "catalog/reporting adapter so paper and hosted supplement tooling "
            "do not import a private research-state surface."
        ),
        "anchor_checks": ANCHOR_CHECKS,
        "delegated_module": "tac.preflight",
        "delegated_check_count": len(check_names),
        "delegated_checks": check_names,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-catalog", action="store_true", help="Print strict-check catalog JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.emit_catalog:
        print(json.dumps(emit_catalog(), indent=2, sort_keys=True))
        return 0
    print(json.dumps({"available": sorted(__all__)}, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ANCHOR_CHECKS",
    "check_42_train_inference_parity",
    "check_dispatch_cli_shell_hazards",
    "check_feature_flags_have_live_objective_effect",
    "check_no_mps_fallback_default",
    "check_public_release_hygiene",
    "check_reverse_engineering_tree_curation",
    "emit_catalog",
    "preflight_all",
]


if __name__ == "__main__":
    raise SystemExit(main())
