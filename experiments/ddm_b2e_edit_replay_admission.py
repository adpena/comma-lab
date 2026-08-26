#!/usr/bin/env python3
"""B2E EDIT-REPLAY admission harness -- the measurable no-fake criterion for burn-2.

THE QUESTION THIS ANSWERS
-------------------------
ns1 measured that all three post-hoc semantic-weight edits on the shipped hv1
vehicle were REFUSED with pose damaged 3.8-5.0x.  The burn-2 regime claims that
training the renderer WITH those edits in the loop makes them nearly free.

This harness is the instrument that decides.  It re-applies the EXACT mp2 edit
constructions to any candidate semantic state and adjudicates the result against a
PRE-REGISTERED bar, so the regime thesis can be refuted by measurement rather than
argued.

PRE-REGISTERED SUCCESS BAR (fixed before any burn-2 run; do not move it)
-----------------------------------------------------------------------
For each edit, define the *pose damage excess* relative to that model's OWN base::

    excess = (pose_edited / pose_base) - 1

The calibration excesses, from the pinned MP2 adjudication, are 2.767 (keep75 minus
keep87), 3.638 (keep87) and 3.959 (mixed q3/q4).  Burn-2 ADMITS an edit when::

    collapse = calibration_excess / burn2_excess >= 50

Anything less and the regime thesis is INSTANCE-refuted for that edit, and this
harness says so.  Comparing each model against its own base is what makes the
comparison fair when burn-2 shifts the base pose.

OBJECT SCOPE
------------
The edits act on the archive member's ``semantic`` section -- the 38-tensor
SemanticTokenRenderer state.  They do NOT act on the ``hpac`` section (the cl1
token model).  See the b2e landing memo: those are two different learned objects
in one archive, and conflating them is what the charter re-pin corrects.

SUBSET BIAS (m96, binding)
--------------------------
Any bounded pair subset is bias-tagged, never reported bare.  Measured on this
corpus, prefix subsets bias the POSE axis 2.5-4.2x HARDER than the population
while biasing seg 0.95-0.97x EASIER.  This harness therefore (a) selects pairs by
seeded STRATIFIED sampling rather than a prefix, and (b) stamps every bounded read
with its bias class.  A bounded read may REFUTE an admission claim; it may never
be the sole basis for granting one.  The n600 advisory leg stays MAIN-fired.

NON-AUTHORITY
-------------
No output here is a score.  Advisory rows are advisory; only the exact contest
evaluator on the exact archive bytes produces a score.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

sd1 = importlib.import_module("experiments.ddm_sd1_semantic_rd_curve")
sm3 = importlib.import_module("experiments.ddm_sm3_semantic_representation")

from tac.pr130_lift.editability_levers import (  # noqa: E402
    FILM_ROW_FAMILY,
    SELECTED_MIXED_Q3_NAMES,
    film_row_order,
    mixed_bit_allocation,
)

SCHEMA = "ddm_b2e_edit_replay_admission.v1"
RATE_DENOMINATOR = 37_545_489

#: The frontier archive the calibration was measured on.
BASE_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_hv1_harvest_compose/ep0634/retained/candidate/archive.zip"
)
BASE_ARCHIVE_BYTES = 182_759
BASE_ARCHIVE_SHA256 = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
BASE_SEMANTIC_STREAM_BYTES = 34_763

#: Pinned calibration -- MP2_ADVISORY_ADJUDICATION.json, sha256
#: 54228227d8d18bbceeca0944371ee2dcb5e773c3135bb1214416247677aeb743.
CALIBRATION: dict[str, dict[str, Any]] = {
    "mixed_q3q4": {
        "mp2_candidate_id": "score_gated_selected_mixed_q3q4",
        "base_pose": 0.00014747,
        "edited_pose": 0.00073123,
        "archive_bytes": 181_936,
        "delta_bytes": -823,
    },
    "film_row_prune_keep87": {
        "mp2_candidate_id": "score_gated_film_row_prune_keep87",
        "base_pose": 0.00014747,
        "edited_pose": 0.0006839,
        "archive_bytes": 182_629,
        "delta_bytes": -130,
    },
    "film_row_prune_keep75_minus_keep87": {
        "mp2_candidate_id": "score_gated_film_row_prune_keep75_minus_keep87",
        "base_pose": 0.00014747,
        "edited_pose": 0.00055551,
        "archive_bytes": 182_734,
        "delta_bytes": -25,
    },
}

#: The charter's pre-registered admission bar.  Do not move without a new charter.
REQUIRED_COLLAPSE_FACTOR = 50.0

EDIT_NAMES = tuple(CALIBRATION)


class B2EError(RuntimeError):
    """Fail-closed error for a b2e replay custody or construction defect."""


# ---------------------------------------------------------------------------
# small io helpers (atomic, payload-retaining)
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _atomic_json(path: Path, value: Any) -> dict[str, Any]:
    return _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# base state loading
# ---------------------------------------------------------------------------


def load_base_state(archive: Path) -> tuple[OrderedDict[str, torch.Tensor], dict[str, Any]]:
    """Load the 38-tensor base semantic state that the archive's section decodes to.

    The hv1 base ``semantic`` section rides the LEGACY q4 parser, and mp2's
    ``expected_state(parser="legacy", ...)`` returns the template unchanged --
    i.e. the retained MZ2 records ARE the decoded base state.  We therefore take
    the state from those records (the same object mp2 and mz2 measured), and
    still bind the archive by size/sha so the provenance names the exact frontier
    bytes this base belongs to.
    """
    import brotli

    mp2 = importlib.import_module("experiments.ddm_mp2_mixed_precision_receiver_close")
    mz2 = importlib.import_module("experiments.ddm_mz2_frozen_section_representation_attack")

    member = mp2.read_stored_member(archive)
    parts = mp2.split_member(member)
    semantic_size = int(parts["semantic_size"])
    raw = brotli.decompress(bytes(parts["semantic"]))

    records, _, _ = mz2._load_records()
    state = OrderedDict(
        (
            record.schema.name,
            torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32)),
        )
        for record in records
    )
    provenance = {
        "source": "archive",
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256_file(archive),
        "semantic_stream_bytes": semantic_size,
        "semantic_raw_bytes": len(raw),
        "semantic_parser": "legacy",
        "state_from": "mz2 retained records (mp2 legacy parser returns the template)",
        "tensor_count": len(state),
    }
    return state, provenance


def load_checkpoint_state(
    checkpoint: Path, template_names: Sequence[str]
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, Any]]:
    """Pull a burn-2 SemanticTokenRenderer state out of a trainer checkpoint.

    Prefers the EMA shadow, because the EMA shadow is what deploys (the EMA
    non-negotiable); falls back to live weights only if no shadow is present, and
    records which one it used.
    """
    with open(checkpoint, "rb") as fh:
        magic = fh.read(4)
    if not (magic.startswith(b"PK\x03\x04") or magic[:1] == b"\x80"):
        raise B2EError(
            f"checkpoint {checkpoint} is not a PyTorch pickle/zip "
            f"(magic {magic!r}); refusing torch.load"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise B2EError(f"checkpoint is not a mapping: {checkpoint}")

    chosen = None
    for key in ("ema_state_dict", "state_dict", "model", "live_state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping) and candidate:
            chosen = (key, candidate)
            break
    if chosen is None:
        raise B2EError(f"checkpoint carries no recognisable state dict: {checkpoint}")

    key, raw_state = chosen
    state = OrderedDict()
    for name in template_names:
        if name not in raw_state:
            raise B2EError(
                f"checkpoint state is not a SemanticTokenRenderer state: missing {name}"
            )
        state[name] = torch.as_tensor(raw_state[name]).detach().cpu().float()
    return state, {
        "source": "checkpoint",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "state_dict_key": key,
        "deployment_weights": payload.get("deployment_weights"),
        "epoch": payload.get("epoch"),
        "tensor_count": len(state),
    }


# ---------------------------------------------------------------------------
# the three EXACT mp2 edit constructions
# ---------------------------------------------------------------------------


def build_mixed_q3q4(state: Mapping[str, torch.Tensor]) -> tuple[bytes, OrderedDict[str, torch.Tensor]]:
    """mp2 ``score_gated_selected_mixed_q3q4``: q3 on the selected set, q4 elsewhere."""
    allocation = OrderedDict(
        (name, bits)
        for name, bits in mixed_bit_allocation(sd1.quantized_names(state)).items()
    )
    return sd1.pack_semantic_state(OrderedDict(state), allocation, legacy_int4=False)


def _prune_rows(
    state: Mapping[str, torch.Tensor], *, drop: str
) -> OrderedDict[str, torch.Tensor]:
    """Zero the FiLM rows a keep-ladder removes.

    ``drop='keep87'`` removes every row outside the top-87% by row L2^2.
    ``drop='keep75_minus_keep87'`` removes only the marginal band that keep87
    retains and keep75 drops -- the differential candidate mp2 built.
    """
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in state.items():
        if name not in FILM_ROW_FAMILY or value.ndim < 2:
            result[name] = value.clone()
            continue
        rows = int(value.shape[0])
        order = film_row_order(value)
        keep87 = max(1, round(rows * 87 / 100.0))
        keep75 = max(1, round(rows * 75 / 100.0))
        reference_keep87 = frozenset(order[:keep87])
        reference_keep75 = frozenset(order[:keep75])
        if not reference_keep75.issubset(reference_keep87):
            raise B2EError(f"non-nested prune reference for {name}")
        if drop == "keep87":
            removed = frozenset(range(rows)) - reference_keep87
        elif drop == "keep75_minus_keep87":
            removed = reference_keep87 - reference_keep75
        else:  # pragma: no cover - guarded by the caller
            raise B2EError(f"unknown prune ladder: {drop}")
        edited = value.clone()
        if removed:
            edited[torch.tensor(sorted(removed), dtype=torch.long)] = 0.0
        result[name] = edited
    return result


def build_edit(
    name: str, state: Mapping[str, torch.Tensor]
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, Any]]:
    """Return the edited state plus a construction record."""
    if name == "mixed_q3q4":
        blob, edited = build_mixed_q3q4(state)
        return edited, {
            "construction": "sd1.pack_semantic_state with mp2 mixed allocation",
            "q3_tensor_names": sorted(SELECTED_MIXED_Q3_NAMES),
            "packed_semantic_raw_bytes": len(blob),
        }
    if name in ("film_row_prune_keep87", "film_row_prune_keep75_minus_keep87"):
        ladder = name.replace("film_row_prune_", "")
        edited = _prune_rows(state, drop=ladder)
        # The shipped receiver stores the surviving rows at standard q4.
        allocation = OrderedDict(
            (tensor, 4) for tensor in sd1.quantized_names(edited)
        )
        blob, quantized = sd1.pack_semantic_state(
            OrderedDict(edited), allocation, legacy_int4=False
        )
        return quantized, {
            "construction": f"FiLM row prune ({ladder}) then standard q4 pack",
            "row_family": sorted(FILM_ROW_FAMILY),
            "packed_semantic_raw_bytes": len(blob),
        }
    raise B2EError(f"unknown edit: {name}")


# ---------------------------------------------------------------------------
# weight-space delta reporting (reproduces the ns1 section-A calibration)
# ---------------------------------------------------------------------------


def weight_space_delta(
    base: Mapping[str, torch.Tensor], edited: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    """Per-tensor and global perturbation report in ns1 section-A units."""
    if set(base) != set(edited):
        raise B2EError("edited state does not have the base tensor set")
    global_sq = 0.0
    delta_sq = 0.0
    rows = []
    for name in base:
        difference = edited[name].float() - base[name].float()
        base_norm = float(base[name].float().square().sum())
        diff_norm = float(difference.square().sum())
        global_sq += base_norm
        delta_sq += diff_norm
        changed = int(torch.count_nonzero(difference))
        rows.append(
            {
                "name": name,
                "elements": int(difference.numel()),
                "changed_elements": changed,
                "changed_fraction": changed / max(int(difference.numel()), 1),
                "delta_l2": diff_norm**0.5,
                "base_l2": base_norm**0.5,
                "relative_l2": (diff_norm**0.5) / max(base_norm**0.5, 1e-12),
                "max_abs": float(difference.abs().max()) if difference.numel() else 0.0,
                "pose_critical": name in FILM_ROW_FAMILY,
            }
        )
    global_l2 = global_sq**0.5
    delta_l2 = delta_sq**0.5
    touched = [row for row in rows if row["changed_elements"] > 0]
    return {
        "global_base_l2": global_l2,
        "delta_l2": delta_l2,
        "relative_l2": delta_l2 / max(global_l2, 1e-12),
        "relative_l2_percent": 100.0 * delta_l2 / max(global_l2, 1e-12),
        "tensors_touched": len(touched),
        "tensors_touched_names": sorted(row["name"] for row in touched),
        "pose_critical_touched": sorted(
            row["name"] for row in touched if row["pose_critical"]
        ),
        "per_tensor": rows,
    }


# ---------------------------------------------------------------------------
# bounded pair selection (seeded, stratified -- never a prefix)
# ---------------------------------------------------------------------------


def bounded_pair_ids(*, seed: int, strata: int, pairs_per_stratum: int) -> list[int]:
    """Seeded stratified pair ids, delegating to the shipped sd1 sampler."""
    return sd1.stratified_random_pair_ids(
        seed=seed, strata=strata, pairs_per_stratum=pairs_per_stratum
    )


def bias_tag(pair_count: int) -> dict[str, Any]:
    """The m96 axis law, attached to every bounded read."""
    return {
        "law": "m96 prefix-bias sign inverts between seg and pose",
        "selection": "seeded stratified over 10 equal strata (NOT a prefix)",
        "pairs": pair_count,
        "population": 600,
        "pose_axis_bias_class": (
            "stratified sampling removes the 2.5-4.2x prefix hardness on pose, but a "
            f"{pair_count}-pair read still carries sampling variance"
        ),
        "seg_axis_bias_class": (
            "stratified sampling removes the 0.95-0.97x prefix easiness on seg"
        ),
        "admissible_use": (
            "may REFUTE an admission claim; may NEVER be the sole basis for granting "
            "one -- the n600 advisory leg stays MAIN-fired"
        ),
        "score_claim": False,
    }


# ---------------------------------------------------------------------------
# admission adjudication against the pre-registered bar
# ---------------------------------------------------------------------------


def adjudicate(
    measured: Mapping[str, Mapping[str, float]],
    *,
    required_collapse: float = REQUIRED_COLLAPSE_FACTOR,
) -> dict[str, Any]:
    """Apply the pre-registered bar to measured burn-2 pose numbers.

    ``measured`` maps an edit name to ``{"base_pose": float, "edited_pose": float}``
    (optionally ``delta_bytes``).  Both come from the SAME instrument on the SAME
    pair set, so the ratio is instrument-independent -- which is what makes an
    advisory read usable for a RATIO claim even though it is never a score.
    """
    rows = {}
    for name, calibration in CALIBRATION.items():
        calibration_excess = calibration["edited_pose"] / calibration["base_pose"] - 1.0
        row: dict[str, Any] = {
            "mp2_candidate_id": calibration["mp2_candidate_id"],
            "calibration_base_pose": calibration["base_pose"],
            "calibration_edited_pose": calibration["edited_pose"],
            "calibration_excess": calibration_excess,
            "calibration_delta_bytes": calibration["delta_bytes"],
            "required_collapse_factor": required_collapse,
        }
        entry = measured.get(name)
        if entry is None:
            row["status"] = "NOT_MEASURED"
            row["verdict"] = "PENDING"
            rows[name] = row
            continue

        base_pose = float(entry["base_pose"])
        edited_pose = float(entry["edited_pose"])
        if base_pose <= 0.0:
            raise B2EError(f"{name}: base_pose must be positive")
        burn_excess = edited_pose / base_pose - 1.0
        row.update(
            {
                "status": "MEASURED",
                "burn2_base_pose": base_pose,
                "burn2_edited_pose": edited_pose,
                "burn2_excess": burn_excess,
            }
        )
        if burn_excess <= 0.0:
            # The edit did not damage pose at all (or helped): unbounded collapse.
            row["collapse_factor"] = float("inf")
            row["verdict"] = "ADMITTED"
        else:
            collapse = calibration_excess / burn_excess
            row["collapse_factor"] = collapse
            row["verdict"] = "ADMITTED" if collapse >= required_collapse else "REFUSED"
        rows[name] = row

    measured_rows = [row for row in rows.values() if row["status"] == "MEASURED"]
    admitted = [row for row in measured_rows if row["verdict"] == "ADMITTED"]
    if not measured_rows:
        overall = "PENDING_MEASUREMENT"
    elif len(admitted) == len(measured_rows):
        overall = "REGIME_THESIS_SUPPORTED"
    elif admitted:
        overall = "REGIME_THESIS_PARTIAL"
    else:
        overall = "REGIME_THESIS_INSTANCE_REFUTED"
    return {
        "edits": rows,
        "overall": overall,
        "measured_count": len(measured_rows),
        "admitted_count": len(admitted),
        "bar": (
            "pose damage excess must collapse >= "
            f"{required_collapse}x vs the pinned mp2 calibration"
        ),
        "score_claim": False,
    }


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_replay(args: argparse.Namespace) -> dict[str, Any]:
    if args.checkpoint is not None:
        base_state, base_provenance = load_base_state(args.base_archive)
        state, provenance = load_checkpoint_state(args.checkpoint, list(base_state))
        provenance["template_from"] = base_provenance
    else:
        state, provenance = load_base_state(args.base_archive)

    results = {}
    artifacts = []
    for name in args.edits:
        edited, construction = build_edit(name, state)
        delta = weight_space_delta(state, edited)
        results[name] = {"construction": construction, "weight_space_delta": delta}
        if args.out_dir is not None:
            # ALWAYS KEEP THE PAYLOAD: the edited state is a real payload, not a
            # scalar, so it is persisted rather than measured-and-discarded.
            target = args.out_dir / "retained" / name
            for index, (tensor_name, value) in enumerate(edited.items()):
                safe = tensor_name.replace(".", "_")
                path = target / f"{index:02d}_{safe}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                buffer = value.contiguous().numpy()
                np.save(path, buffer)
                artifacts.append({"path": str(path), "sha256": _sha256_file(path)})

    return {
        "schema": SCHEMA,
        "stage": "replay",
        "generated_utc": _utc(),
        "axis": "[byte-only scorer-free weight-space replay]",
        "score_claim": False,
        "promotion_eligible": False,
        "object_scope": (
            "archive 'semantic' section == 38-tensor SemanticTokenRenderer; NOT the "
            "'hpac' section (cl1 token model)"
        ),
        "source": provenance,
        "edits": results,
        "retained_artifacts": artifacts,
    }


def stage_pairs(args: argparse.Namespace) -> dict[str, Any]:
    ids = bounded_pair_ids(
        seed=args.seed, strata=args.strata, pairs_per_stratum=args.pairs_per_stratum
    )
    return {
        "schema": SCHEMA,
        "stage": "pairs",
        "generated_utc": _utc(),
        "seed": args.seed,
        "strata": args.strata,
        "pairs_per_stratum": args.pairs_per_stratum,
        "pair_ids": ids,
        "bias_tag": bias_tag(len(ids)),
    }


def stage_admit(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(args.measured.read_text(encoding="utf-8"))
    measured = payload.get("measured", payload)
    if not isinstance(measured, Mapping):
        raise B2EError("--measured must contain a mapping of edit -> pose row")
    verdict = adjudicate(measured, required_collapse=args.required_collapse)
    return {
        "schema": SCHEMA,
        "stage": "admit",
        "generated_utc": _utc(),
        "axis": payload.get("axis", "[unlabelled advisory -- caller must tag the axis]"),
        "score_claim": False,
        "promotion_eligible": False,
        "measured_input": str(args.measured),
        "measured_input_sha256": _sha256_file(args.measured),
        "adjudication": verdict,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("replay", "pairs", "admit"))
    parser.add_argument("--base-archive", type=Path, default=BASE_ARCHIVE)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="burn-2 SemanticTokenRenderer checkpoint; omit to replay on the base",
    )
    parser.add_argument(
        "--edits",
        nargs="+",
        choices=EDIT_NAMES,
        default=list(EDIT_NAMES),
    )
    parser.add_argument("--out-dir", type=Path, help="SSD retention root")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--strata", type=int, default=8)
    parser.add_argument("--pairs-per-stratum", type=int, default=4)
    parser.add_argument("--measured", type=Path, help="admit: measured pose rows JSON")
    parser.add_argument("--required-collapse", type=float, default=REQUIRED_COLLAPSE_FACTOR)
    parser.add_argument("--verify-base-pins", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.verify_base_pins:
        if args.base_archive.stat().st_size != BASE_ARCHIVE_BYTES:
            raise B2EError("base archive byte count differs from the pinned frontier")
        if _sha256_file(args.base_archive) != BASE_ARCHIVE_SHA256:
            raise B2EError("base archive sha256 differs from the pinned frontier")
    if args.stage == "admit" and args.measured is None:
        raise B2EError("--measured is required for the admit stage")

    if args.stage == "replay":
        result = stage_replay(args)
    elif args.stage == "pairs":
        result = stage_pairs(args)
    else:
        result = stage_admit(args)

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(args.out_dir / f"B2E_{args.stage.upper()}_RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
