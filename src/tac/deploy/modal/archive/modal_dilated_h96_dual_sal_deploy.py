"""Retired Modal dilated h96 dual-saliency launcher.

This compatibility stub exists so historical imports fail closed instead of
silently dispatching the retired ``train_tac`` path. Use the canonical provider
bundle and active Modal T1 actuators for score-lowering work.
"""

from __future__ import annotations

RETIREMENT_MESSAGE = (
    "modal_dilated_h96_dual_sal_deploy.py is retired. Use "
    "`python -m tac.deploy.build_bundle` for provider-neutral bundles or "
    "`experiments/modal_t1_balle_endtoend.py` for the active T1 Ballé Modal path."
)


def main() -> None:
    raise SystemExit(RETIREMENT_MESSAGE)


if __name__ == "__main__":
    main()
