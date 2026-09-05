#!/usr/bin/env python
"""ddm_md3 — build ALTERNATE born-vehicle STARTING STATES and their step-0 probe configs.

Why this module exists (MEASURED, and it corrects the charter's premise):
``experiments/ddm_qbr1_born_fairform_burn_prep.build_initial_state`` takes the r10 stage-03
checkpoint's ``ema.shadow`` **verbatim and takes no seed at all**.  The born vehicle's random
initialisation lives far upstream --- ``ddm_qbflow_packet.initialize_params(SEED)`` with
``SEED = 20260827`` in ``experiments/ddm_qbflow_rate_first_rung.py`` --- behind the whole
qbt1 r1..r10 training chain.  There is therefore **no init seed in any burn cell's config** to
flip, and a different RANDOM initialisation cannot be produced inside this arm's budget.

What CAN be produced, legitimately and without touching a single weight by hand, is a ladder of
different STARTING POINTS: every state below is a state some governed run actually reached and
retained.  ``md1``'s instrument reads exactly one field from a cell config ---
``config["initial_state"]["path"]`` (``ddm_md1_micro_to_macro.py:431``) --- so pointing that field
at one of these states and sweeping with ``--max-step 0`` measures that start's **step-0 wrong
pool** for one forward pass.  That is the free ceiling gate on the whole different-start question:
if starts far apart in weight space still paint the same wrong pool, the pool is data-anchored and
no Metal cell can move it.

Sources (all retained, all produced by training, none hand-edited):

* ``r10 stage_03_end`` ``live_state_dict``  --- same checkpoint as the incumbent, different basis
* ``r9 / r8 / r7 / r6 stage_03_end`` ``ema.shadow`` --- other governed runs that reached the same
  stage on their own trajectories
* ``r10 stage_03`` periodic checkpoints --- earlier points on the incumbent's own trajectory
* ``packet.initialize_params(seed)`` --- the ROOT seeded generic init.  This one is a **diagnostic
  null anchor only**: it is untrained, and the born gate
  (``birth_gate_revision``: "inherited born start; raw r5 balanced-CE start is forbidden") forbids
  it as a cell start.  It is built only to put the pool-overlap scale on a floor.

Every emitted state carries the same ``ddm_qbt2b_initialized_qbf1_state.v1`` schema and the same
strict tensor-key round trip that ``build_initial_state`` enforces, plus provenance naming the
source file, stage, step and basis.  Writes are atomic (``qbt.atomic_torch``).

$0.  CPU only.  Reads retained checkpoints; writes only under ``--store``.  No score claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_qbflow_packet as packet
from experiments import ddm_qbr1_born_fairform_burn_prep as qbr1
from experiments import ddm_qbt1_qbflow_trainer as qbt

SCHEMA_STATE = "ddm_qbt2b_initialized_qbf1_state.v1"
SCHEMA_INDEX = "ddm_md3_alternate_initial_states.v1"
AXIS = (
    "[macOS-CPU advisory; alternate born-vehicle starting states built from retained governed-run "
    "checkpoints; not contest authority; no score claim]"
)

QBT1_ROOT = Path(
    "/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer"
)
STAGE_03 = "stage_03_joint_boundary_interior_birth"
STAGE_03_END = "stage_03_joint_boundary_interior_birth_end"

# The incumbent every burn cell starts from, for the record.
INCUMBENT = qbr1.QBR_INITIAL_STATE

# (candidate_id, run, basis)  --- the ladder.  `basis` is either "ema_shadow" or "live".
STAGE_END_CANDIDATES: tuple[tuple[str, str, str], ...] = (
    ("r10_live", "governed_n32_r10", "live"),
    ("r9_shadow", "governed_n32_r9", "ema_shadow"),
    ("r8_shadow", "governed_n32_r8", "ema_shadow"),
    ("r7_shadow", "governed_n32_r7", "ema_shadow"),
    ("r6_shadow", "governed_n32_r6", "ema_shadow"),
)


class MD3Error(RuntimeError):
    """ddm_md3 refuses rather than emitting an unfaithful or hand-edited start."""


# ---------------------------------------------------------------------------
def _reference_keys() -> set[str]:
    """The exact QBF1 twin's tensor key set --- build_initial_state's own gate."""

    return set(qbt.load_initial_model(torch.device("cpu")).state_dict())


def _extract(checkpoint: dict[str, Any], basis: str) -> dict[str, torch.Tensor]:
    if basis == "ema_shadow":
        state = checkpoint.get("ema", {}).get("shadow", {})
    elif basis == "live":
        state = checkpoint.get("live_state_dict", {})
    else:
        raise MD3Error(f"unknown basis {basis!r}")
    if not state:
        raise MD3Error(f"checkpoint carries no {basis} state")
    return state


def _write_state(
    store: Path,
    candidate_id: str,
    state: dict[str, torch.Tensor],
    provenance: dict[str, Any],
    reference: set[str],
) -> dict[str, Any]:
    if set(state) != reference:
        raise MD3Error(f"{candidate_id}: tensor set differs from the exact QBF1 twin")
    payload = {
        "schema": SCHEMA_STATE,
        "state_dict": {name: value.detach().cpu().clone().float() for name, value in state.items()},
        "provenance": provenance,
    }
    path = store / "alternate_initial_states" / f"md3_{candidate_id}_state.pt"
    fact = qbt.atomic_torch(path, payload)
    roundtrip = torch.load(path, map_location="cpu", weights_only=False)
    if set(roundtrip["state_dict"]) != reference:
        raise MD3Error(f"{candidate_id}: state failed strict tensor-key round trip")
    return fact


def _probe_config(store: Path, candidate_id: str, fact: dict[str, Any]) -> dict[str, Any]:
    """The minimal config md1's instrument reads: only ``initial_state.path`` is consumed."""

    config = {
        "schema": "ddm_md3_step0_probe_config.v1",
        "candidate_id": candidate_id,
        "purpose": (
            "step-0 wrong-pool probe only; md1's sweep reads config['initial_state']['path'] and "
            "nothing else (ddm_md1_micro_to_macro.py:431). NOT an authorized training config."
        ),
        "launch_authorized": False,
        "initial_state": dict(fact),
    }
    path = store / "probe_configs" / f"md3_{candidate_id}_probe.json"
    qbt.atomic_json(path, config)
    return {"config_path": str(path.resolve()), "candidate_id": candidate_id}


# ---------------------------------------------------------------------------
def _relative_distance(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, float]:
    keys = sorted(a)
    sq = sum(float(((a[k].float() - b[k].float()) ** 2).sum()) for k in keys)
    norm = sum(float((a[k].float() ** 2).sum()) for k in keys)
    d = float(np.sqrt(sq))
    n = float(np.sqrt(norm))
    return {"l2_distance": d, "reference_l2_norm": n, "relative_l2": (d / n) if n else float("nan")}


def build(args: argparse.Namespace) -> dict[str, Any]:
    store = Path(args.store)
    reference = _reference_keys()

    incumbent_payload = torch.load(INCUMBENT, map_location="cpu", weights_only=False)
    incumbent_state = {k: v.float() for k, v in incumbent_payload["state_dict"].items()}
    if set(incumbent_state) != reference:
        raise MD3Error("the incumbent initial state no longer matches the QBF1 twin")

    rows: list[dict[str, Any]] = []
    for candidate_id, run, basis in STAGE_END_CANDIDATES:
        source = QBT1_ROOT / run / STAGE_03 / "checkpoints" / "stage_03_end.pt"
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != STAGE_03_END:
            raise MD3Error(f"{candidate_id}: source is not the stage-03 end checkpoint")
        state = _extract(checkpoint, basis)
        provenance = {
            "source_checkpoint": qbt.file_fact(source),
            "source_run": run,
            "source_stage": checkpoint["stage"],
            "source_step": int(checkpoint["step"]),
            "basis": basis,
            "produced_by": "training; copied verbatim, never hand-edited",
            "root_init_seed_shared_with_incumbent": True,
            "note": (
                "a different STARTING POINT on the QBF1 lineage, NOT a different random "
                "initialisation: every governed run descends from "
                "ddm_qbflow_packet.initialize_params(20260827)"
            ),
        }
        fact = _write_state(store, candidate_id, state, provenance, reference)
        probe = _probe_config(store, candidate_id, fact)
        row = {
            "candidate_id": candidate_id,
            "state": fact,
            "probe": probe,
            "provenance": provenance,
            "distance_from_incumbent": _relative_distance(incumbent_state, {k: v.float() for k, v in state.items()}),
            "born_gate_eligible": True,
        }
        rows.append(row)
        del checkpoint, state

    # Earlier points on the incumbent's OWN trajectory.
    for step in args.r10_periodic_steps:
        candidate_id = f"r10_periodic_{step:06d}_shadow"
        source = QBT1_ROOT / "governed_n32_r10" / STAGE_03 / "checkpoints" / f"periodic_step_{step:06d}.pt"
        if not source.is_file():
            raise MD3Error(f"{candidate_id}: retained periodic checkpoint absent: {source}")
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        state = _extract(checkpoint, "ema_shadow")
        provenance = {
            "source_checkpoint": qbt.file_fact(source),
            "source_run": "governed_n32_r10",
            "source_stage": checkpoint.get("stage"),
            "source_step": int(checkpoint["step"]),
            "basis": "ema_shadow",
            "produced_by": "training; copied verbatim, never hand-edited",
            "root_init_seed_shared_with_incumbent": True,
            "note": "an earlier point on the INCUMBENT's own trajectory",
        }
        fact = _write_state(store, candidate_id, state, provenance, reference)
        probe = _probe_config(store, candidate_id, fact)
        rows.append(
            {
                "candidate_id": candidate_id,
                "state": fact,
                "probe": probe,
                "provenance": provenance,
                "distance_from_incumbent": _relative_distance(
                    incumbent_state, {k: v.float() for k, v in state.items()}
                ),
                "born_gate_eligible": True,
            }
        )
        del checkpoint, state

    # The ROOT seeded generic init --- diagnostic null anchor, born-gate INELIGIBLE.
    if args.root_init_seeds:
        model = qbt.load_initial_model(torch.device("cpu"))
        template = model.state_dict()
        for seed in args.root_init_seeds:
            candidate_id = f"root_init_seed_{seed}"
            params = packet.initialize_params(int(seed))
            boundary, interior = packet.initialize_latents(int(seed), qbt.N)
            fresh = qbt.QBFLOWTorch(params, boundary, interior)
            state = {name: value.detach().cpu().clone().float() for name, value in fresh.state_dict().items()}
            if set(state) != set(template):
                raise MD3Error(f"{candidate_id}: fresh init tensor set differs from the QBF1 twin")
            provenance = {
                "source": "ddm_qbflow_packet.initialize_params / initialize_latents",
                "root_init_seed": int(seed),
                "basis": "seeded_generic_initialization",
                "produced_by": "deterministic seeded builder; never hand-edited",
                "root_init_seed_shared_with_incumbent": int(seed) == 20260827,
                "born_gate": (
                    "INELIGIBLE as a cell start: untrained. The QBR1 birth gate requires an "
                    "inherited born start ('raw r5 balanced-CE start is forbidden'). Built as a "
                    "DIAGNOSTIC null anchor for the pool-overlap scale only."
                ),
            }
            fact = _write_state(store, candidate_id, state, provenance, set(template))
            probe = _probe_config(store, candidate_id, fact)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "state": fact,
                    "probe": probe,
                    "provenance": provenance,
                    "distance_from_incumbent": _relative_distance(incumbent_state, state),
                    "born_gate_eligible": False,
                }
            )

    index = {
        "schema": SCHEMA_INDEX,
        "axis": AXIS,
        "incumbent": {
            "path": str(INCUMBENT.resolve()),
            "fact": qbt.file_fact(INCUMBENT),
            "builder": (
                "ddm_qbr1_born_fairform_burn_prep.build_initial_state -- r10 stage-03 EMA shadow, "
                "verbatim, NO SEED"
            ),
            "root_init_seed": 20260827,
            "root_init_source": "experiments/ddm_qbflow_rate_first_rung.py::SEED",
        },
        "candidates": rows,
        "scope_note": (
            "every candidate is a different STARTING POINT within the QBF1 lineage from root seed "
            "20260827; none is a different random initialisation. Any verdict drawn from these "
            "carries that scope."
        ),
    }
    qbt.atomic_json(store / "ALTERNATE_INITIAL_STATES.json", index)
    return index


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", required=True, help="the only directory written")
    parser.add_argument(
        "--r10-periodic-steps",
        type=int,
        nargs="*",
        default=[],
        metavar="STEP",
        help="earlier points on the incumbent's own r10 stage-03 trajectory",
    )
    parser.add_argument(
        "--root-init-seeds",
        type=int,
        nargs="*",
        default=[],
        metavar="SEED",
        help="seeded generic inits; DIAGNOSTIC null anchors only (born-gate ineligible)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    index = build(args)
    summary = {
        "incumbent_sha256": index["incumbent"]["fact"]["sha256"],
        "candidates": [
            {
                "candidate_id": row["candidate_id"],
                "sha256": row["state"]["sha256"],
                "relative_l2_from_incumbent": row["distance_from_incumbent"]["relative_l2"],
                "born_gate_eligible": row["born_gate_eligible"],
                "probe_config": row["probe"]["config_path"],
            }
            for row in index["candidates"]
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
