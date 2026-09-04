"""The C2 ``IntegerPlaneEmitter`` DSL lever, bound to the trainer that OWNS its flags.

Why this module exists (ddm_ql3, 2026-09-04). The lever lived in ``curriculum_dsl.py``, and the
registry grades a module's emitted flags against **that module's** trainer
(``lever_registry.module_trainer_paths``). ``curriculum_dsl`` declares
``TRAINER_RELPATHS = (levelset entry point, the base it imports its primitives from)``, so this
lever's three flags were graded against a pair of trainers that never declared them — and
``completeness().stale`` reported them as DSL drift. MEASURED at HEAD before the move:

    stale == ['--integer-plane-emitter-basis', '--integer-plane-emitter-mode',
              '--integer-plane-emitter-policy-sha256']

That report was a FALSE positive of the drift class it exists to catch: the flags are real and
live, declared by the dedicated C2 parser at
``src/tac/boundary_math/integer_plane_banded_trainer.py`` (``build_parser``, the three
``add_argument("--integer-plane-emitter-…")`` lines), and ``git log -S`` finds no commit in which
the level-set trainer ever carried them. The lever was never wrong; its **binding** was
undeclarable, because a module-level ``TRAINER_RELPATH`` is the only way to state one and
``curriculum_dsl`` legitimately needs the plural form for its own levers.

The cure is the registry's OWN mechanism, not a side table: give the lever a module that declares
its trainer. This is deliberately NOT a hand-typed exception list beside the DSL (CLAUDE.md,
"never build a parallel registry beside the DSL") — the binding below is read by the same AST
scan every other lever module goes through, so a future flag rename still fails loudly.

``curriculum_dsl`` re-exports :func:`IntegerPlaneEmitter`, so every historical import path
(``from tac.witness_dsl.curriculum_dsl import IntegerPlaneEmitter``) keeps working unchanged.

Import direction is one-way ON PURPOSE: nothing here imports ``curriculum_dsl`` at module scope
(``Lever`` is imported inside the factory, as the policy symbols already were), so
``curriculum_dsl`` can import this module at the top of its own import block without a cycle.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tac.witness_dsl.curriculum_dsl import Lever
    from tac.witness_dsl.integer_plane_emitter_policy import IntegerPlaneEmitterPolicy

# The trainer whose argparse OWNS every flag this module emits. Declared, not defaulted: the
# registry cannot otherwise tell an intentional binding from an author who never considered the
# question (``lever_registry.module_declares_trainer``), and that indistinguishability is exactly
# what produced the false ``stale`` report above.
TRAINER_RELPATH = "src/tac/boundary_math/integer_plane_banded_trainer.py"


def IntegerPlaneEmitter(*, policy: IntegerPlaneEmitterPolicy) -> Lever:
    """Typed C2 policy lever for the dedicated band trainer.

    The compatibility mode is deliberately argv-inert. ``BANDED_TRAINING``
    emits only flags owned by the dedicated C2 parser; it is therefore not
    composable into the level-set ``BASELINE`` program.
    """

    from tac.witness_dsl.curriculum_dsl import Lever
    from tac.witness_dsl.integer_plane_emitter_policy import (
        BANDED_TRAINING_RECEIPT_SCHEMA,
        POLICY_CONTRACT_RECEIPT_KEY,
        IntegerPlaneEmitterPolicy,
        PolicyMode,
    )

    if not isinstance(policy, IntegerPlaneEmitterPolicy):
        raise TypeError("IntegerPlaneEmitter requires an IntegerPlaneEmitterPolicy")
    contract = policy.compile_contract()
    active = policy.mode is PolicyMode.BANDED_TRAINING
    overrides = (
        {
            "--integer-plane-emitter-mode": policy.mode.value,
            "--integer-plane-emitter-basis": policy.basis.value,
            "--integer-plane-emitter-policy-sha256": contract["policy_sha256"],
        }
        if active
        else {}
    )
    receipts = (
        {"--integer-plane-emitter-policy-sha256": BANDED_TRAINING_RECEIPT_SCHEMA}
        if active
        else {}
    )
    return Lever(
        "IntegerPlaneEmitter",
        overrides=overrides,
        epochs_delta=0,
        notes=(
            ("argv-effective dedicated C2 band trainer; " if active else "argv-inert default-OFF C2 emitter; ")
            + "basis="
            f"{contract['basis']}; policy_sha256={contract['policy_sha256']}; "
            "launch/score/promotion/pointer authority sealed false"
        ),
        lawrefs={},
        constant_manifest={},
        runtime_receipt_schemas=receipts,
        policy_contracts={POLICY_CONTRACT_RECEIPT_KEY: contract},
    )


__all__ = ["TRAINER_RELPATH", "IntegerPlaneEmitter"]
