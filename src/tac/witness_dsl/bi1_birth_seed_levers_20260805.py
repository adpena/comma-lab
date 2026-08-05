# SPDX-License-Identifier: MIT
"""ddm_bi1 (#924) TR1 BIRTH seed/amplify levers.

BI1 builds the half p4x explicitly left OWED: a TR1-native seed/amplify path for
born components.  It is not a port of the retired ``--seed-islands`` flags.  The
trainer consumes new ``--tr1-birth-*`` runtime flags, defaults them OFF, and keeps
them args-only so an unarmed run preserves TR1Config/config_hash/checkpoint bytes.

The mechanism is intentionally scorer-free: GT ``lstars`` define Lane/Movable
birth support through the existing eased island geometry, the token lattice is
initialized on those supports, and an anchor term keeps the support from being
washed out.  No SegNet/PoseNet forward and no quality verdict live here.
"""

from __future__ import annotations

from tac.witness_dsl.curriculum_dsl import Lever

TRAINER_RELPATH = "experiments/train_tr1_partition_renderer_mlx.py"

BI1_SEED_WEIGHT_START = 0.35
BI1_AMPLIFY_WEIGHT_START = 0.05


def _manifest(flag: str, value, rung: str, provenance: str) -> dict:
    return {flag: {"value": value, "rung": rung, "provenance": provenance}}


def lever_tr1_birth_seed_amplify(
    *,
    seed_weight: float = BI1_SEED_WEIGHT_START,
    amplify_weight: float = BI1_AMPLIFY_WEIGHT_START,
    classes: str = "lane",
    dilate_px: int = 1,
    persist: str = "inverse_thickness",
) -> Lever:
    """TR1 seed/amplify BIRTH path, Lane super-nuclei first.

    Falsifier: a scorer-free smoke shows the flag is ON but seeded token cells
    are absent or immediately unanchored, in which case BI1 is an implementation
    failure before any scorer A/B is admissible.  A later hinge A/B null would
    close only this seed/amplify formulation, never the birth-completion family.
    """
    if float(seed_weight) <= 0.0:
        raise ValueError("seed_weight must be > 0; 0.0 is the trainer OFF control")
    if float(amplify_weight) < 0.0:
        raise ValueError("amplify_weight must be >= 0")
    names = tuple(s.strip().lower() for s in str(classes).split(",") if s.strip())
    if not names or any(n not in {"lane", "movable"} for n in names):
        raise ValueError("classes must be a comma list drawn from lane,movable")
    if int(dilate_px) < 0:
        raise ValueError("dilate_px must be >= 0")
    if persist not in {"uniform", "inverse_thickness"}:
        raise ValueError("persist must be uniform|inverse_thickness")
    class_token = "_".join(names)
    return Lever(
        name=f"tr1_birth_seed_{class_token}_w{seed_weight:g}_a{amplify_weight:g}",
        overrides={
            "--tr1-birth-seed-weight": str(seed_weight),
            "--tr1-birth-seed-classes": ",".join(names),
            "--tr1-birth-seed-dilate-px": int(dilate_px),
            "--tr1-birth-amplify-weight": str(amplify_weight),
            "--tr1-birth-amplify-persist": persist,
        },
        notes="BI1 #924 TR1-native birth seed/amplify path. Seeds GT Lane/Movable "
              "birth supports into the token lattice and anchors them with a scorer-free "
              "token term. Default OFF is byte-identical; ON is A/B-ready but carries "
              "score_claim=False until an owned hinge A/B runs.",
        constant_manifest={
            **_manifest("--tr1-birth-seed-weight", float(seed_weight),
                        "RACED-NOT-ASSERTED",
                        "start value is a bounded token-lattice initialization amplitude, "
                        "not a measured optimum; must sweep before any adopt/kill verdict"),
            **_manifest("--tr1-birth-amplify-weight", float(amplify_weight),
                        "RACED-NOT-ASSERTED",
                        "small anchor weight starts below the seg objective scale and only "
                        "prevents immediate erasure of the seeded token support"),
            **_manifest("--tr1-birth-seed-classes", ",".join(names),
                        "MEASURED-PRIOR FROM p4x",
                        "p4x birth matrix measured Lane and Movable as the real annihilation "
                        "classes; Lane first is the super-nucleus rung BI1 owns"),
            **_manifest("--tr1-birth-seed-dilate-px", int(dilate_px),
                        "REUSED GEOMETRY",
                        "one-pixel eased support is the existing island_protection geometry; "
                        "this lever does not assert it is the own optimum"),
            **_manifest("--tr1-birth-amplify-persist", persist,
                        "REUSED GEOMETRY",
                        "inverse_thickness is the existing persistence weighting for the "
                        "lowest-persistence island pixels"),
        },
        runtime_receipt_schemas={
            "tr1_birth_seed_init": "ON-only telemetry event in train_tr1_partition_renderer_mlx.py"
        },
        policy_contracts={"score_claim": False, "default_off_byte_identity": True},
    )
