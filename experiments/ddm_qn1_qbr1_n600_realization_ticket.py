# SPDX-License-Identifier: MIT
"""QN1 — the QBR1 CONDITIONAL-N600-BUY made fireable in one command.

The QBR1 six-cell fair-form burn ends in a pre-registered adjudication
(``ddm_qbr1_adjudication_result.v1``).  Its ``CONDITIONAL-N600-BUY`` row says:
build a *same-object retained n600 ticket* rather than transferring the n32
Horvitz-Thompson estimate or BR2's old distortion.  This module is that ticket
generator.

``ticket`` reads the sealed adjudication, refuses unless the typed outcome is
``OPTIMIZATION_LIVE_DISTORTION_ROUTE``, selects the winning treatment cell,
binds that cell's step-5000 exact QBF1 archive bytes and its receiver-decoded
field by SHA-256, and writes a sealed n600 realization fire order in the BR2
protocol (``ddm_br2_born_object_scorer_realization.py``: 30-pair restartable
chunks, every render/scorer payload retained, resume-from the output root, an
explicit scorer claim, explicit launch authorization).

``realize`` is that BR2 protocol with the object taken from the ticket instead
of from frozen module constants.  BR2 and QXR1 each hardcode a single archive
SHA and a single output root, so neither can consume the burn's field; this
subcommand delegates the scorer-touching core to BR2's own measured
``realize_chunk`` and only parameterises the object, the output root, and the
rate term.  Nothing here runs a scorer, Modal, or Metal: ``ticket`` is
scorer-free by construction and ``realize`` is MAIN-owned.

Every number is advisory.  The n32 S_hat values quoted in a ticket are the
adjudication's own selection statistics; they are NEVER transferred into the
n600 row, which is measured fresh from the bound archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tarfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _root in (REPO, SRC):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments import ddm_qbflow_packet as qbf1

SCHEMA = "ddm_qn1_n600_realization_fire_order.v1"
DRY_RUN_SCHEMA = "ddm_qn1_dry_run_receipt.v1"
REALIZED_SCHEMA = "ddm_qn1_n600_realized_result.v1"

# --- burn custody (read-only; the burn owns these paths) ------------------------------------------
AP_BURN_ROOT = Path("/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure")
ADJUDICATION_RESULT = AP_BURN_ROOT / "ADJUDICATION_RESULT.json"
ADJUDICATION_SCHEMA_PATH = AP_BURN_ROOT / "ADJUDICATION_SCHEMA.json"
BURN_RUN_ROOT = AP_BURN_ROOT / "runs"
DRY_RUN_CELL_ROOT = BURN_RUN_ROOT / "seed_20260902/control_native100"

# --- QN1 custody ----------------------------------------------------------------------------------
OUTPUT_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_qn1_qbr1_n600_realization_ticket")
REALIZATION_ROOT = Path("/Volumes/APDataStore/pact/ddm_qn1_qbr1_n600_realization")

# --- pre-registered adjudication vocabulary (ddm_qbr1_preregistered_adjudication.v1) ---------------
ADJUDICATION_SCHEMA_ID = "ddm_qbr1_adjudication_result.v1"
RESULT_SCHEMA_ID = "ddm_qbr1_fairform_result.v1"
MILESTONE_SCHEMA_ID = "ddm_qbr1_realized_milestone.v1"
LIVE_OUTCOME = "OPTIMIZATION_LIVE_DISTORTION_ROUTE"
CLOSED_OUTCOME = "OPTIMIZATION_CLOSED_CHANGED_CAPACITY_OBJECT_ONLY"
MIXED_OUTCOME = "INCONCLUSIVE_MIXED_NO_FAMILY_CLOSURE"
TREATMENT_ARM = "treatment_zero_native"
CONTROL_ARM = "control_native100"
ENDPOINT_STEP = 5_000

# --- BR2 realization protocol (experiments/ddm_br2_born_object_scorer_realization.py) --------------
BR2_RUNNER = REPO / "experiments/ddm_br2_born_object_scorer_realization.py"
N = 600
H, W = 384, 512  # BR2/QBT1 render geometry; asserted against br2.H/br2.W at fire time
CHUNK_PAIRS = 30
RATE_DENOMINATOR = 37_545_489
# MEASURED, BR2 REALIZED_RESULT.json: 20 retained chunks summing 1,058,094,084 B in 479.663 s.
BR2_MEASURED_RETAINED_BYTES = 1_058_094_084
BR2_MEASURED_ELAPSED_SECONDS = 479.6629898548126
# MEASURED, the burn's own REDERIVED_TIMING.json br2_realization_seconds_each.
QBR1_TIMING_REALIZATION_SECONDS_EACH = 484.769
EXPECTED_WALL_SECONDS = 485
# DERIVED: 1.25 x BR2's measured retention + 100 MB headroom = 1,422,617,605 B, rounded up.
MINIMUM_FREE_BYTES = 1_500_000_000

# --- score law + pre-registered falsifier ----------------------------------------------------------
TARGET_SCORE = 0.12
FALSIFIER_D_SEG = 0.01
# memory m110: pose absolute budget <= 1.25e-4 (dS/d_d_pose 626.5 at the operating point).
FALSIFIER_D_POSE = 1.25e-4
# CLAUDE.md goal banner (refreshed 2026-09-01): the rate corner demands archive <= 137,986 B.
BYTE_FEASIBLE_CEILING = 137_986
AFR1_SCORE = 0.14797617125559104
AFR1_ARCHIVE_BYTES = 180_002

# Archives that already carry a realized scorer row.  QXR1's "identical-by-construction => derive,
# don't fire" rule: refuse to buy an n600 row for bytes that were already realized.
SCORED_ANCESTOR_SHA256: Mapping[str, str] = {
    "br2_born_object": "0e2ffdfaa5fe481d481dd70a9672a67f80b9aad7648f0c775fe2956dd3a4841d",
    "qxr1_qxo1_object": "2487f5150fd3c38087fb5ada48d00e953c7d88a8a7219e29fbf53420657bb07f",
    "afr1_canonical_frontier": "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25",
}

# --- scorer claim contract (BR2's, with the lane prefix parameterised to this arm) ------------------
ACTIVE_CLAIMS = REPO / ".omx/state/active_lane_dispatch_claims.md"
CLAIM_PREFIX = "ddm_qn1_"
CLAIM_PLACEHOLDER = "QN1_SCORER_CLAIM_ID"


class QN1Error(RuntimeError):
    """Fail-closed refusal for QN1 adjudication, selection, binding, or custody violations."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise QN1Error(f"required file is absent: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def tar_member_bytes(tar: tarfile.TarFile, name: str, *, container: Path) -> bytes:
    """Fail closed on an absent or non-regular container member instead of leaking a KeyError."""
    try:
        stream = tar.extractfile(name)
    except KeyError as exc:
        raise QN1Error(f"retained container member is absent: {name!r} in {container}") from exc
    if stream is None:
        raise QN1Error(f"retained container member is unreadable: {name!r} in {container}")
    return stream.read()


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise QN1Error(f"required JSON is absent: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_scorer_claim_id(claim_id: str) -> str:
    """Reject the placeholder and any non-QN1 lane id before a ticket is written."""
    if claim_id == CLAIM_PLACEHOLDER or not claim_id.strip():
        raise QN1Error(
            f"scorer claim is still the placeholder {CLAIM_PLACEHOLDER!r}; "
            "MAIN must bind a unique live local_macos_cpu claim id before the ticket can be fired"
        )
    if not claim_id.startswith(CLAIM_PREFIX):
        raise QN1Error(f"scorer claim must be a QN1-owned lane id starting with {CLAIM_PREFIX!r}: {claim_id!r}")
    return claim_id


def assert_active_scorer_claim(claim_id: str) -> dict[str, Any]:
    """BR2's active-claim contract with the lane prefix parameterised to QN1.

    BR2 hardcodes the ``ddm_br2_`` prefix, so its function cannot admit a QN1 lane; the row
    parsing, the newest-row-wins rule, and the 24 h conflicting-scorer scan are identical.
    """
    assert_scorer_claim_id(claim_id)
    rows: list[dict[str, str]] = []
    for line in ACTIVE_CLAIMS.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) == 8 and fields[0].startswith("20"):
            rows.append(
                {
                    "timestamp": fields[0],
                    "lane_id": fields[2],
                    "platform": fields[3],
                    "status": fields[6],
                    "raw": line,
                }
            )
    newest_by_lane: dict[str, dict[str, str]] = {}
    for row in rows:
        newest_by_lane.setdefault(row["lane_id"], row)
    own = newest_by_lane.get(claim_id)
    if own is None or own["platform"] != "local_macos_cpu" or not own["status"].startswith("active_"):
        raise QN1Error("newest QN1 row must be an active local_macos_cpu scorer claim")
    cutoff = datetime.now(UTC) - timedelta(hours=24)
    conflicts = []
    for lane_id, row in newest_by_lane.items():
        if lane_id == claim_id or "scorer" not in lane_id or not row["status"].startswith("active_"):
            continue
        if datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")) >= cutoff:
            conflicts.append(row["raw"])
    if conflicts:
        raise QN1Error(f"another live scorer claim remains active: {conflicts}")
    return {"claim_id": claim_id, "registry": file_fact(ACTIVE_CLAIMS), "row": own["raw"]}


def chunk_plan(*, n: int = N, chunk_pairs: int = CHUNK_PAIRS) -> list[dict[str, int]]:
    """The BR2 restartable chunk plan: contiguous 30-pair spans covering exactly [0, n)."""
    if n <= 0 or chunk_pairs <= 0:
        raise QN1Error("chunk plan needs a positive population and chunk size")
    plan = []
    for index, start in enumerate(range(0, n, chunk_pairs)):
        stop = min(n, start + chunk_pairs)
        plan.append(
            {
                "index": index,
                "first_pair": start,
                "last_pair": stop - 1,
                "pairs": stop - start,
                "payload_name": f"scorer_pairs_{start:04d}_{stop - 1:04d}.npz",
            }
        )
    if sum(row["pairs"] for row in plan) != n:
        raise QN1Error("chunk plan does not partition the population")
    return plan


def score_law(*, d_seg: float, d_pose: float, archive_bytes: int) -> dict[str, Any]:
    """S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37,545,489, recomputed from components."""
    if archive_bytes <= 0:
        raise QN1Error("score law needs a positive archive byte count")
    if d_seg < 0.0 or d_pose < 0.0:
        raise QN1Error("score law needs non-negative distortion components")
    seg_term = 100.0 * d_seg
    pose_term = math.sqrt(10.0 * d_pose)
    rate = 25.0 * archive_bytes / RATE_DENOMINATOR
    distortion_budget = TARGET_SCORE - rate
    return {
        "formula": "100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489",
        "rate_denominator": RATE_DENOMINATOR,
        "archive_bytes": int(archive_bytes),
        "d_seg": d_seg,
        "d_pose": d_pose,
        "seg_term": seg_term,
        "pose_term": pose_term,
        "distortion": seg_term + pose_term,
        "rate": rate,
        "S": seg_term + pose_term + rate,
        "sub_0_12_distortion_budget": distortion_budget,
        "sub_0_12_d_seg_budget_after_measured_pose": (distortion_budget - pose_term) / 100.0,
        "byte_feasible_ceiling": BYTE_FEASIBLE_CEILING,
        "byte_feasible": int(archive_bytes) <= BYTE_FEASIBLE_CEILING,
        "delta_vs_0_12": seg_term + pose_term + rate - TARGET_SCORE,
        "delta_vs_afr1": seg_term + pose_term + rate - AFR1_SCORE,
    }


def falsifier_row(*, archive_bytes: int) -> dict[str, Any]:
    """The pre-registered QXR1 falsifier, re-derived against the bound archive."""
    opened = score_law(d_seg=FALSIFIER_D_SEG, d_pose=FALSIFIER_D_POSE, archive_bytes=archive_bytes)
    return {
        "prediction": (
            "the burn-trained field realizes n600 distortion far above the sub-0.12 corner; "
            "BR2's old born object realized d_seg 0.17077688 and d_pose 115.837 on the same protocol"
        ),
        "falsifier": (
            f"realized d_seg <= {FALSIFIER_D_SEG} AND d_pose <= {FALSIFIER_D_POSE} on this exact "
            f"archive binding at <= {BYTE_FEASIBLE_CEILING} bytes opens the first byte-feasible "
            "distortion path (source: QXR1 fire order ddm_qxr1_qxo1_scorer_fire_order.v1)"
        ),
        "falsifier_d_seg": FALSIFIER_D_SEG,
        "falsifier_d_pose": FALSIFIER_D_POSE,
        "falsifier_byte_ceiling": BYTE_FEASIBLE_CEILING,
        "falsifier_S_if_exactly_met": opened["S"],
        # DERIVED, and deliberately loud: meeting the QXR1 falsifier does NOT by itself clear 0.12.
        # At d_seg exactly 0.01 the seg term alone is 1.0.  The falsifier marks a REGIME change
        # (a byte-feasible object whose distortion is finally in reach), not the target.
        "falsifier_alone_clears_0_12": opened["S"] < TARGET_SCORE,
        "falsifier_is_a_regime_marker_not_the_target": True,
        "d_seg_required_for_0_12_at_the_falsifier_pose": (
            TARGET_SCORE - opened["rate"] - opened["pose_term"]
        )
        / 100.0,
        "falsifier_d_seg_over_required_d_seg": FALSIFIER_D_SEG
        / ((TARGET_SCORE - opened["rate"] - opened["pose_term"]) / 100.0),
        "status": "UNTESTED_UNTIL_MAIN_SCORER_FIRE",
        "no_distortion_transfer": True,
    }


# --------------------------------------------------------------------------------------------------
# object binding
# --------------------------------------------------------------------------------------------------


def bind_object(
    milestone: Mapping[str, Any],
    *,
    scored_ancestors: Mapping[str, str] = SCORED_ANCESTOR_SHA256,
) -> dict[str, Any]:
    """Bind a milestone's exact QBF1 archive bytes and its receiver-decoded field by SHA-256."""
    if milestone.get("schema") != MILESTONE_SCHEMA_ID:
        raise QN1Error(f"milestone schema drifted: {milestone.get('schema')!r} != {MILESTONE_SCHEMA_ID!r}")
    reencode = milestone.get("reencode")
    if not isinstance(reencode, Mapping):
        raise QN1Error("milestone carries no reencode block")
    archive_row = reencode["archive"]
    packet_row = reencode["packet"]
    container = Path(archive_row["container"])
    container_fact = file_fact(container)
    with tarfile.open(container, mode="r") as tar:
        archive = tar_member_bytes(tar, archive_row["container_member"], container=container)
        packet = tar_member_bytes(tar, packet_row["container_member"], container=container)
    if len(archive) != int(archive_row["bytes"]) or sha256_bytes(archive) != archive_row["sha256"]:
        raise QN1Error("bound archive bytes or SHA-256 differ from the milestone record")
    if len(packet) != int(packet_row["bytes"]) or sha256_bytes(packet) != packet_row["sha256"]:
        raise QN1Error("bound packet bytes or SHA-256 differ from the milestone record")
    if int(milestone["archive_bytes_exact"]) != len(archive):
        raise QN1Error("milestone archive_bytes_exact disagrees with the retained archive")
    recomputed_rate = 25.0 * len(archive) / RATE_DENOMINATOR
    if not math.isclose(float(milestone["rate_exact"]), recomputed_rate, rel_tol=0.0, abs_tol=1e-15):
        raise QN1Error("milestone rate_exact is not the score law applied to the retained archive")
    receiver_packet = qbf1.read_deterministic_archive(archive)
    if receiver_packet != packet:
        raise QN1Error("archive receiver output is not bit-identical to the retained packet.qbf")
    if qbf1.deterministic_archive(receiver_packet) != archive:
        raise QN1Error("receiver decode/re-encode did not reconstruct archive.zip bit-identically")
    decoded = qbf1.decode_packet(receiver_packet)
    expected_sections = {
        qbf1.SECTION_CONFIG,
        qbf1.SECTION_MODEL,
        qbf1.SECTION_LATENT_META,
        qbf1.SECTION_LATENTS,
    }
    if set(decoded.sections) != expected_sections:
        raise QN1Error("receiver-decoded QBF section set differs from the QBF1 contract")
    records = qbf1.decode_latent_table(decoded.sections[qbf1.SECTION_LATENTS])
    if set(records) != set(range(N)):
        raise QN1Error("receiver-decoded archive does not carry all 600 latent records")
    section_digests = {str(key): sha256_bytes(value) for key, value in sorted(decoded.sections.items())}
    field_digest = sha256_bytes(json.dumps(section_digests, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    archive_digest = sha256_bytes(archive)
    for name, digest in scored_ancestors.items():
        if archive_digest == digest:
            raise QN1Error(
                f"bound archive is byte-identical to the already-realized ancestor {name!r}; "
                "derive the row from that realization instead of buying it again"
            )
    return {
        "container": container_fact,
        "archive": {
            "container_member": archive_row["container_member"],
            "bytes": len(archive),
            "sha256": archive_digest,
        },
        "packet": {
            "container_member": packet_row["container_member"],
            "bytes": len(packet),
            "sha256": packet_row["sha256"],
        },
        "decoded_field_digest": field_digest,
        "decoded_section_digests": section_digests,
        "latent_records": len(records),
        "receiver_packet_bit_identity": True,
        "receiver_archive_roundtrip_bit_identity": True,
        "scored_ancestor_sha256": dict(scored_ancestors),
        "byte_identical_to_scored_ancestor": False,
        "rate_exact_recomputed": recomputed_rate,
    }


# --------------------------------------------------------------------------------------------------
# adjudication consumption + winner selection
# --------------------------------------------------------------------------------------------------


def select_winner(adjudication: Mapping[str, Any]) -> dict[str, Any]:
    """Select the treatment cell that both won its seed and passed its own pose corner.

    The pre-registered family outcome needs >= 2 of 3 seeds; a single n600 buy is one cell, so that
    cell must itself satisfy both legs.  Ties break on the lowest S_hat, then the lowest seed.
    """
    if adjudication.get("schema") != ADJUDICATION_SCHEMA_ID:
        raise QN1Error(f"adjudication schema drifted: {adjudication.get('schema')!r} != {ADJUDICATION_SCHEMA_ID!r}")
    disposition = adjudication.get("disposition")
    if disposition != LIVE_OUTCOME:
        raise QN1Error(
            f"adjudication disposition is {disposition!r}; the n600 buy fires only on {LIVE_OUTCOME!r} "
            f"(the pre-registered alternatives {CLOSED_OUTCOME!r} and {MIXED_OUTCOME!r} carry no buy)"
        )
    seed_rows = adjudication.get("seed_rows")
    if not isinstance(seed_rows, list) or not seed_rows:
        raise QN1Error("adjudication carries no seed rows")
    eligible = [
        row for row in seed_rows if bool(row.get("treatment_win")) and bool(row.get("treatment_pose_corner_pass"))
    ]
    if not eligible:
        raise QN1Error(
            "no treatment cell both won its seed and passed its own pose corner; "
            "the family outcome is LIVE but no single cell is buyable"
        )
    winner = min(eligible, key=lambda row: (float(row["treatment_S_hat"]), int(row["seed"])))
    return {
        "seed": int(winner["seed"]),
        "arm": TREATMENT_ARM,
        "cell_id": f"seed_{int(winner['seed'])}_{TREATMENT_ARM}",
        "treatment_S_hat_n32": float(winner["treatment_S_hat"]),
        "control_S_hat_n32": float(winner["control_S_hat"]),
        "delta_treatment_minus_control_n32": float(winner["delta_treatment_minus_control"]),
        "treatment_win": True,
        "treatment_pose_corner_pass": True,
        "eligible_seeds": [int(row["seed"]) for row in eligible],
        "selection_rule": "lowest treatment S_hat among seeds that won AND passed their own pose corner",
        "n32_numbers_are_selection_statistics_only": True,
    }


def assert_preregistration(path: Path) -> dict[str, Any]:
    """Re-derive the burn's own pre-registered adjudication schema instead of trusting this module."""
    schema = read_json(path)
    if schema.get("schema") != "ddm_qbr1_preregistered_adjudication.v1":
        raise QN1Error(f"pre-registered adjudication schema drifted: {schema.get('schema')!r}")
    if int(schema.get("population_n", -1)) != N:
        raise QN1Error(f"pre-registration population_n {schema.get('population_n')!r} is not the n600 buy")
    if int(schema.get("endpoint_step", -1)) != ENDPOINT_STEP:
        raise QN1Error(f"pre-registration endpoint_step {schema.get('endpoint_step')!r} != {ENDPOINT_STEP}")
    if schema.get("no_n600_buy_before_sign_repeats") is not True:
        raise QN1Error("pre-registration no longer carries the no_n600_buy_before_sign_repeats rule")
    return {**schema, "fact": file_fact(path)}


def resolve_winner_result(adjudication: Mapping[str, Any], winner: Mapping[str, Any]) -> dict[str, Any]:
    """Locate the winning cell's RESULT.json from the adjudication's own recorded source facts."""
    needle = f"/seed_{winner['seed']}/{TREATMENT_ARM}/"
    matches = [row for row in adjudication.get("source_results", []) if needle in str(row.get("path", ""))]
    if len(matches) != 1:
        raise QN1Error(f"adjudication source_results does not name exactly one {winner['cell_id']} result")
    recorded = matches[0]
    live = file_fact(Path(recorded["path"]))
    if live["sha256"] != recorded["sha256"] or live["bytes"] != recorded["bytes"]:
        raise QN1Error(f"winning cell RESULT.json drifted since adjudication: {recorded['path']}")
    return live


def milestone_from_result(result: Mapping[str, Any], *, endpoint_step: int = ENDPOINT_STEP) -> Mapping[str, Any]:
    if result.get("schema") != RESULT_SCHEMA_ID:
        raise QN1Error(f"cell result schema drifted: {result.get('schema')!r} != {RESULT_SCHEMA_ID!r}")
    if result.get("complete") is not True:
        raise QN1Error("winning cell result is not complete; the endpoint milestone is not final")
    milestones = result.get("milestones")
    if not isinstance(milestones, list) or not milestones:
        raise QN1Error("winning cell result carries no milestones")
    endpoint = milestones[-1]
    if int(endpoint.get("step", -1)) != endpoint_step:
        raise QN1Error(f"winning cell endpoint milestone is step {endpoint.get('step')!r}, not {endpoint_step}")
    return endpoint


# --------------------------------------------------------------------------------------------------
# fire order
# --------------------------------------------------------------------------------------------------


def realization_argv(
    *,
    runner: Path,
    output: Path,
    ticket_path: Path,
    claim_id: str,
    python: Path,
) -> list[str]:
    """Verbatim argv for the BR2-protocol realization, using only flags this module defines."""
    return [
        str(python),
        str(runner),
        "realize",
        "--ticket",
        str(ticket_path),
        "--output",
        str(output),
        "--resume-from",
        str(output),
        "--scorer-claim-id",
        claim_id,
        "--launch-authorized",
    ]


def build_ticket(
    *,
    adjudication: Mapping[str, Any],
    adjudication_fact: Mapping[str, Any] | None,
    winner: Mapping[str, Any],
    milestone: Mapping[str, Any],
    result_fact: Mapping[str, Any] | None,
    preregistration: Mapping[str, Any] | None,
    claim_id: str,
    realization_output: Path,
    ticket_path: Path,
    dry_run: bool,
    dry_run_notes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    obj = bind_object(milestone)
    archive_bytes = int(obj["archive"]["bytes"])
    plan = chunk_plan()
    argv = realization_argv(
        runner=Path(__file__).resolve(),
        output=realization_output,
        ticket_path=ticket_path,
        claim_id=claim_id,
        python=REPO / ".venv/bin/python",
    )
    ticket: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory n600 realization; not contest CPU/CUDA authority]",
        "score_claim": False,
        "promotable": False,
        "pointer_moved": False,
        "mode": "DRY_RUN_PLUMBING" if dry_run else "LIVE",
        "disposition": "DRY_RUN_NOT_FIREABLE" if dry_run else "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN n600 local scorer-realization scheduler",
        "adjudication": {
            "schema": adjudication.get("schema"),
            "disposition": adjudication.get("disposition"),
            "treatment_wins": adjudication.get("treatment_wins"),
            "treatment_pose_corner_passes": adjudication.get("treatment_pose_corner_passes"),
            "seed_rows": list(adjudication.get("seed_rows", [])),
            "source_fact": dict(adjudication_fact) if adjudication_fact else None,
            "preregistration": dict(preregistration) if preregistration else None,
            "synthetic": bool(dry_run),
        },
        "winner": {
            **dict(winner),
            "endpoint_step": int(milestone.get("step", -1)),
            "milestone_selection_n": milestone.get("selection_n"),
            "milestone_population_n": milestone.get("population_n"),
            "milestone_d_seg_hat_n32": milestone.get("d_seg_hat"),
            "milestone_d_pose_hat_n32": milestone.get("d_pose_hat"),
            "milestone_S_hat_n32": milestone.get("S_hat"),
            "milestone_rate_exact": milestone.get("rate_exact"),
            "milestone_pose_corner_pass": milestone.get("pose_corner_pass"),
            "result_fact": dict(result_fact) if result_fact else None,
        },
        "object": obj,
        "realization": {
            "protocol": "ddm_br2_born_object_scorer_realization.v1",
            "protocol_source": file_fact(BR2_RUNNER),
            "runner": file_fact(Path(__file__).resolve()),
            "subcommand": "realize",
            "population_n": N,
            "chunk_pairs": CHUNK_PAIRS,
            "chunks": len(plan),
            "chunk_plan": plan,
            "output_root": str(realization_output),
            "resume_from": str(realization_output),
            "retention": (
                "every render chunk (camera pair, SegNet logits/argmax, PoseNet pose6, both targets), "
                "the receiver-decoded inputs, the per-pair rows, and the result are retained; "
                "no payload is deleted by the runner"
            ),
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
            "minimum_free_bytes_derivation": (
                f"1.25 x BR2's measured {BR2_MEASURED_RETAINED_BYTES} B n600 retention + 100 MB headroom "
                "= 1,422,617,605 B, rounded up to 1.5 GB"
            ),
            "expected_wall_seconds": EXPECTED_WALL_SECONDS,
            "expected_wall_seconds_sources": {
                "br2_realized_result_elapsed_seconds": BR2_MEASURED_ELAPSED_SECONDS,
                "qbr1_rederived_timing_seconds_each": QBR1_TIMING_REALIZATION_SECONDS_EACH,
            },
            "argv": argv,
            "command": " ".join(["OMP_NUM_THREADS=8", "MKL_NUM_THREADS=8", *argv]),
            "scorer_claim_id": claim_id,
        },
        "score_law": score_law(d_seg=0.0, d_pose=0.0, archive_bytes=archive_bytes),
        "prediction": falsifier_row(archive_bytes=archive_bytes),
        "fire_trigger": (
            "the sealed adjudication disposition is OPTIMIZATION_LIVE_DISTORTION_ROUTE, MAIN appends a fresh "
            f"unique active local_macos_cpu claim {claim_id!r} with no newer active scorer claim within 24 h, "
            f"AP free bytes >= {MINIMUM_FREE_BYTES}, and the runner re-matches archive "
            f"{obj['archive']['sha256']} plus decoded field {obj['decoded_field_digest']} at fire time"
        ),
        "no_distortion_transfer": True,
        "n32_advisory_numbers_are_not_transferred_to_n600": True,
        "scorer_invocations_by_this_generator": 0,
        "metal_invocations_by_this_generator": 0,
        "modal_invocations_by_this_generator": 0,
        "contest_eval_invocations_by_this_generator": 0,
        "boundaries": [
            "advisory macOS CPU row; not contest CPU/CUDA authority",
            "the n32 stratified HT S_hat values are selection statistics, never the n600 result",
            "BR2's DISTORTION-REFUSED row (d_seg 0.17077688, d_pose 115.837) belongs to the OLD born object",
            "a negative n600 verdict, if any, is INSTANCE scope on this exact archive binding",
            "no contest inflate runtime tree is sealed; the receiver is locally byte-closed only",
        ],
    }
    if dry_run:
        ticket["dry_run"] = dict(dry_run_notes or {})
    return ticket


# --------------------------------------------------------------------------------------------------
# CLI actions
# --------------------------------------------------------------------------------------------------


def assert_not_burn_custody(path: Path, *, what: str) -> Path:
    """Nothing QN1 names as a write target may sit inside the live burn's custody tree."""
    resolved = path.resolve()
    burn = AP_BURN_ROOT.resolve()
    if resolved == burn or burn in resolved.parents:
        raise QN1Error(f"{what} may never sit inside the live burn custody root: {resolved}")
    return resolved


def assert_output_root(output: Path) -> Path:
    resolved = assert_not_burn_custody(output, what="the QN1 ticket output root")
    allowed = OUTPUT_ROOT.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise QN1Error(f"QN1 may only write under {allowed}: refused {resolved}")
    return resolved


def latest_milestone_path(cell_root: Path) -> Path:
    paths = sorted((cell_root / "milestones").glob("step_*/MILESTONE.json"))
    if not paths:
        raise QN1Error(f"no milestone is materialized under {cell_root}")
    return paths[-1]


def synthetic_adjudication(
    *,
    milestone: Mapping[str, Any],
    cell_root: Path,
    result_path: Path,
    force_win: bool,
    force_pose_pass: bool,
    disposition: str = LIVE_OUTCOME,
) -> dict[str, Any]:
    """A synthetic adjudication over an existing milestone, for plumbing proofs only."""
    seed = int(cell_root.parent.name.split("_", 1)[1])
    s_hat = float(milestone["S_hat"])
    return {
        "schema": ADJUDICATION_SCHEMA_ID,
        "axis": "[SYNTHETIC plumbing stand-in; not a burn adjudication]",
        "score_claim": False,
        "synthetic": True,
        "disposition": disposition,
        "treatment_wins": 3 if force_win else 0,
        "treatment_pose_corner_passes": 3 if force_pose_pass else 0,
        "seed_rows": [
            {
                "seed": seed,
                "control_S_hat": s_hat,
                "treatment_S_hat": s_hat,
                "delta_treatment_minus_control": 0.0,
                "treatment_win": force_win,
                "treatment_pose_corner_pass": force_pose_pass,
            }
        ],
        "source_results": [{"path": str(result_path), "bytes": 0, "sha256": "0" * 64}],
    }


def _refusal(label: str, thunk: Any) -> dict[str, Any]:
    try:
        thunk()
    except QN1Error as exc:
        return {"refusal": label, "fired": True, "message": str(exc)}
    return {"refusal": label, "fired": False, "message": "DID NOT REFUSE — falsifier hit"}


def dry_run(*, output: Path, cell_root: Path, milestone_path: Path | None) -> dict[str, Any]:
    resolved_output = assert_output_root(output)
    resolved_output.mkdir(parents=True, exist_ok=True)
    path = milestone_path or latest_milestone_path(cell_root)
    milestone = read_json(path)
    result_path = cell_root / "RESULT.json"

    # (a) refusal suite over synthetic negatives, using the cell's REAL flags where they exist.
    real_pose_pass = bool(milestone.get("pose_corner_pass"))
    refusals = [
        _refusal(
            "missing_adjudication_result",
            lambda: read_json(AP_BURN_ROOT / "ADJUDICATION_RESULT_DOES_NOT_EXIST.json"),
        ),
        _refusal(
            "outcome_inconclusive_mixed",
            lambda: select_winner(
                synthetic_adjudication(
                    milestone=milestone,
                    cell_root=cell_root,
                    result_path=result_path,
                    force_win=True,
                    force_pose_pass=True,
                    disposition=MIXED_OUTCOME,
                )
            ),
        ),
        _refusal(
            "outcome_optimization_closed",
            lambda: select_winner(
                synthetic_adjudication(
                    milestone=milestone,
                    cell_root=cell_root,
                    result_path=result_path,
                    force_win=False,
                    force_pose_pass=False,
                    disposition=CLOSED_OUTCOME,
                )
            ),
        ),
        _refusal(
            "pose_corner_failed_real_cell_flag",
            lambda: select_winner(
                synthetic_adjudication(
                    milestone=milestone,
                    cell_root=cell_root,
                    result_path=result_path,
                    force_win=True,
                    force_pose_pass=real_pose_pass,
                )
            ),
        ),
        _refusal("scorer_claim_placeholder", lambda: assert_scorer_claim_id(CLAIM_PLACEHOLDER)),
        _refusal("scorer_claim_wrong_lane_prefix", lambda: assert_scorer_claim_id("ddm_br2_scorer_20260903")),
        _refusal(
            "incomplete_cell_result",
            lambda: milestone_from_result({"schema": RESULT_SCHEMA_ID, "complete": False, "milestones": [milestone]}),
        ),
        # Deterministic regardless of which milestone is bound: a step that is never the endpoint.
        _refusal(
            "endpoint_step_is_not_5000",
            lambda: milestone_from_result(
                {
                    "schema": RESULT_SCHEMA_ID,
                    "complete": True,
                    "milestones": [{**milestone, "step": ENDPOINT_STEP - 1}],
                },
            ),
        ),
        _refusal(
            "output_root_inside_live_burn_custody",
            lambda: assert_output_root(AP_BURN_ROOT / "runs"),
        ),
        _refusal(
            "output_root_outside_qn1_custody",
            lambda: assert_output_root(Path("/Volumes/VertigoDataTier/pact/some_other_arm")),
        ),
        _refusal(
            "realization_root_inside_live_burn_custody",
            lambda: assert_not_burn_custody(
                AP_BURN_ROOT / "runs", what="the QN1 realization output root"
            ),
        ),
        _refusal(
            "preregistration_schema_drift",
            lambda: assert_preregistration(AP_BURN_ROOT / "BUILD_RECEIPT.json"),
        ),
        _refusal(
            "archive_sha_drift_vs_milestone_record",
            lambda: bind_object(
                {
                    **milestone,
                    "reencode": {
                        **milestone["reencode"],
                        "archive": {
                            **milestone["reencode"]["archive"],
                            "sha256": SCORED_ANCESTOR_SHA256["br2_born_object"],
                        },
                    },
                }
            ),
        ),
        # The ancestor guard compares the RECOMPUTED archive digest, so the probe declares this
        # cell's own real digest as a stand-in ancestor rather than editing the milestone record.
        _refusal(
            "byte_identical_to_scored_ancestor",
            lambda: bind_object(
                milestone,
                scored_ancestors={"probe_stand_in_ancestor": milestone["reencode"]["archive"]["sha256"]},
            ),
        ),
    ]

    # (b) binding proof: the REAL archive/field/chunk plan, under an explicitly synthetic adjudication.
    adjudication = synthetic_adjudication(
        milestone=milestone,
        cell_root=cell_root,
        result_path=result_path,
        force_win=True,
        force_pose_pass=True,
    )
    winner = select_winner(adjudication)
    ticket_path = resolved_output / "DRY_RUN_FIRE_ORDER.json"
    ticket = build_ticket(
        adjudication=adjudication,
        adjudication_fact=None,
        winner=winner,
        milestone=milestone,
        result_fact=None,
        preregistration=None,
        claim_id=f"{CLAIM_PREFIX}scorer_dryrun_placeholder_not_for_fire",
        realization_output=REALIZATION_ROOT,
        ticket_path=ticket_path,
        dry_run=True,
        dry_run_notes={
            "plumbing_only": True,
            "cell_used": str(cell_root),
            "cell_arm": CONTROL_ARM,
            "cell_arm_is_a_control_not_a_treatment": True,
            "milestone_path": str(path),
            "milestone_step": milestone.get("step"),
            "milestone_is_the_5000_endpoint": int(milestone.get("step", -1)) == ENDPOINT_STEP,
            "observed_real_pose_corner_pass": real_pose_pass,
            "synthetic_pose_corner_pass_used_for_binding": True,
            "warning": (
                "this run proves the archive/field binding, the chunk plan, and the refusal paths ONLY. "
                "The cell is the CONTROL arm, the milestone is not the step-5000 endpoint, and the "
                "adjudication flags are synthetic. Nothing here is a treatment verdict or a score."
            ),
        },
    )
    ticket_written = write_json(ticket_path, ticket)
    receipt = {
        "schema": DRY_RUN_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU scorer-free plumbing proof]",
        "score_claim": False,
        "promotable": False,
        "plumbing_only": True,
        "cell_arm_is_a_control_not_a_treatment": True,
        "milestone": file_fact(path),
        "milestone_step": milestone.get("step"),
        "observed_real_milestone_rows": {
            "S_hat": milestone.get("S_hat"),
            "d_seg_hat": milestone.get("d_seg_hat"),
            "d_pose_hat": milestone.get("d_pose_hat"),
            "rate_exact": milestone.get("rate_exact"),
            "archive_bytes_exact": milestone.get("archive_bytes_exact"),
            "pose_corner_pass": milestone.get("pose_corner_pass"),
            "selection_n": milestone.get("selection_n"),
            "population_n": milestone.get("population_n"),
        },
        "binding": ticket["object"],
        "chunk_plan_chunks": ticket["realization"]["chunks"],
        "chunk_plan_first": ticket["realization"]["chunk_plan"][0],
        "chunk_plan_last": ticket["realization"]["chunk_plan"][-1],
        "score_law_at_zero_distortion": ticket["score_law"],
        "prediction": ticket["prediction"],
        "refusals": refusals,
        "refusals_fired": sum(1 for row in refusals if row["fired"]),
        "refusals_total": len(refusals),
        "all_refusals_fired": all(row["fired"] for row in refusals),
        "ticket": ticket_written,
        "scorer_invocations": 0,
        "metal_invocations": 0,
        "modal_invocations": 0,
        "contest_eval_invocations": 0,
        "wrote_only_under": str(resolved_output),
    }
    write_json(resolved_output / "DRY_RUN_RECEIPT.json", receipt)
    return receipt


def write_json(path: Path, payload: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return file_fact(path)


def ticket_action(
    *,
    adjudication_path: Path,
    output: Path,
    claim_id: str,
    realization_output: Path,
) -> dict[str, Any]:
    resolved_output = assert_output_root(output)
    realization_output = assert_not_burn_custody(realization_output, what="the QN1 realization output root")
    assert_scorer_claim_id(claim_id)
    preregistration = assert_preregistration(ADJUDICATION_SCHEMA_PATH)
    adjudication = read_json(adjudication_path)
    adjudication_fact = file_fact(adjudication_path)
    winner = select_winner(adjudication)
    result_fact = resolve_winner_result(adjudication, winner)
    milestone = milestone_from_result(
        read_json(Path(result_fact["path"])), endpoint_step=int(preregistration["endpoint_step"])
    )
    if bool(milestone.get("pose_corner_pass")) is not bool(winner["treatment_pose_corner_pass"]):
        raise QN1Error(
            "the winning cell's endpoint milestone pose_corner_pass disagrees with the adjudication seed row; "
            "the adjudication is stale or the run store drifted"
        )
    ticket_path = resolved_output / "FIRE_ORDER.json"
    ticket = build_ticket(
        adjudication=adjudication,
        adjudication_fact=adjudication_fact,
        winner=winner,
        milestone=milestone,
        result_fact=result_fact,
        preregistration=preregistration,
        claim_id=claim_id,
        realization_output=realization_output,
        ticket_path=ticket_path,
        dry_run=False,
    )
    write_json(ticket_path, ticket)
    return ticket


# --------------------------------------------------------------------------------------------------
# realize — the BR2 protocol with the object taken from the ticket
# --------------------------------------------------------------------------------------------------


def storage_preflight(output: Path, *, required: int = MINIMUM_FREE_BYTES) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    usage = os.statvfs(output)
    free = int(usage.f_bavail * usage.f_frsize)
    if free < required:
        raise QN1Error(f"storage preflight refused: free={free} required={required}")
    return {
        "root": str(output.resolve()),
        "free_bytes": free,
        "required_free_bytes": required,
        "status": "PASS",
        "cleanup": "certify-or-block; no retained QN1 payload may be deleted by this runner",
    }


def aggregate(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    archive_bytes: int,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """BR2's aggregation with the rate term bound to the ticket's archive instead of a constant."""
    seg_errors = sum(int(row["seg_errors"]) for row in pair_rows)
    seg_pixels = sum(int(row["seg_pixels"]) for row in pair_rows)
    pose_sse = sum(float(row["pose_squared_error_sum"]) for row in pair_rows)
    pose_values = sum(int(row["pose_values"]) for row in pair_rows)
    if seg_pixels <= 0:
        raise QN1Error("aggregation received no scorer pixels")
    class_rows = []
    for class_id, class_name in enumerate(class_names):
        target_pixels = sum(int(row["per_class"][class_id]["target_pixels"]) for row in pair_rows)
        errors = sum(int(row["per_class"][class_id]["errors"]) for row in pair_rows)
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "target_pixels": target_pixels,
                "errors": errors,
                "conditional_d_seg": None if target_pixels == 0 else errors / target_pixels,
                "contribution_to_global_d_seg": errors / seg_pixels,
            }
        )
    if seg_pixels != N * H * W or pose_values != N * 6:
        raise QN1Error("global scorer denominators differ from n600")
    if sum(row["target_pixels"] for row in class_rows) != seg_pixels:
        raise QN1Error("per-class target-pixel denominator does not partition global pixels")
    if sum(row["errors"] for row in class_rows) != seg_errors:
        raise QN1Error("per-class errors do not partition global SegNet errors")
    components = score_law(
        d_seg=seg_errors / seg_pixels,
        d_pose=pose_sse / pose_values,
        archive_bytes=archive_bytes,
    )
    components["per_class"] = class_rows
    components["seg_errors"] = seg_errors
    components["seg_pixels"] = seg_pixels
    components["pose_squared_error_sum"] = pose_sse
    components["pose_values"] = pose_values
    return components


def realize(
    *,
    ticket_path: Path,
    output: Path,
    resume_from: Path,
    claim_id: str,
    launch_authorized: bool,
) -> dict[str, Any]:
    """MAIN-only: run/resume the ticket's n600 realization through BR2's measured chunk core."""
    if not launch_authorized:
        raise QN1Error("n600 scorer realization requires explicit launch authorization")
    if resume_from.resolve() != output.resolve():
        raise QN1Error("--resume-from must name the exact QN1 realization output root")
    assert_not_burn_custody(output, what="the QN1 realization output root")
    ticket = read_json(ticket_path)
    if ticket.get("schema") != SCHEMA:
        raise QN1Error(f"ticket schema drifted: {ticket.get('schema')!r} != {SCHEMA!r}")
    if ticket.get("mode") != "LIVE" or ticket.get("disposition") != "QUEUED-WITH-A-FIRE-ORDER":
        raise QN1Error("only a LIVE, queued QN1 fire order may be realized; dry-run tickets are not fireable")
    if ticket["realization"]["scorer_claim_id"] != claim_id:
        raise QN1Error("scorer claim id does not match the sealed ticket")
    if Path(ticket["realization"]["output_root"]).resolve() != output.resolve():
        raise QN1Error(
            f"--output {output.resolve()} is not the output root sealed in the ticket "
            f"({ticket['realization']['output_root']})"
        )
    claim = assert_active_scorer_claim(claim_id)

    import numpy as np
    import torch

    from experiments import ddm_br2_born_object_scorer_realization as br2
    from experiments import ddm_qbt1_qbflow_trainer as qbt1
    from experiments import ddm_qbz1_descent_rate_configuration as qbz1
    from tac.gt_lineage import AUTHORITY_LINEAGE, assert_gt_lineage
    from tac.scorer import load_differentiable_scorers

    obj = ticket["object"]
    storage = storage_preflight(output)
    container = Path(obj["container"]["path"])
    live_container = file_fact(container)
    if live_container["sha256"] != obj["container"]["sha256"]:
        raise QN1Error("the ticket's retained container drifted since the ticket was sealed")
    with tarfile.open(container, mode="r") as tar:
        archive = tar_member_bytes(tar, obj["archive"]["container_member"], container=container)
    if sha256_bytes(archive) != obj["archive"]["sha256"] or len(archive) != int(obj["archive"]["bytes"]):
        raise QN1Error("archive bytes drifted from the sealed ticket binding")
    packet = qbf1.read_deterministic_archive(archive)
    if sha256_bytes(packet) != obj["packet"]["sha256"]:
        raise QN1Error("receiver-decoded packet drifted from the sealed ticket binding")
    decoded = qbf1.decode_packet(packet)
    section_digests = {str(key): sha256_bytes(value) for key, value in sorted(decoded.sections.items())}
    field_digest = sha256_bytes(json.dumps(section_digests, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    if field_digest != obj["decoded_field_digest"]:
        raise QN1Error("decoded field digest drifted from the sealed ticket binding")
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for name, payload in (("archive.zip", archive), ("packet.qbf", packet)):
        target = inputs / name
        if target.exists():
            if file_fact(target)["sha256"] != sha256_bytes(payload):
                raise QN1Error(f"existing retained input differs; refusing overwrite: {target}")
        else:
            qbt1.atomic_bytes(target, payload)

    assert_gt_lineage(qbz1.GT_ARGMAX, required=AUTHORITY_LINEAGE, instrument="QN1 DALI partition")
    assert_gt_lineage(qbz1.GT_POSE6, required=AUTHORITY_LINEAGE, instrument="QN1 DALI pose")
    gt = np.load(qbz1.GT_ARGMAX, mmap_mode="r", allow_pickle=False)
    pose_target = np.load(qbz1.GT_POSE6, mmap_mode="r", allow_pickle=False)
    if (br2.N, br2.H, br2.W, br2.CHUNK_PAIRS, br2.RATE_DENOMINATOR) != (N, H, W, CHUNK_PAIRS, RATE_DENOMINATOR):
        raise QN1Error("BR2 protocol constants drifted from the QN1 ticket contract")
    if gt.shape != (N, H, W) or gt.dtype != np.uint8 or pose_target.shape != (N, 6):
        raise QN1Error("registered scorer-target geometry differs")
    torch.manual_seed(qbz1.SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model = br2.model_from_packet(packet)
    model.eval()
    posenet, segnet = load_differentiable_scorers(REPO / "upstream", device=torch.device("cpu"))
    posenet.eval()
    segnet.eval()

    started = time.time()
    chunks: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for row in ticket["realization"]["chunk_plan"]:
        ids = list(range(int(row["first_pair"]), int(row["last_pair"]) + 1))
        checkpoint, rows = br2.realize_chunk(
            output,
            ids=ids,
            model=model,
            gt=gt,
            pose_target=pose_target,
            posenet=posenet,
            segnet=segnet,
        )
        chunks.append(checkpoint["payload"])
        pair_rows.extend(rows)
        print(json.dumps({"realized_pairs": ids[-1] + 1, "n": N, "resumed": checkpoint["resumed"]}), flush=True)
    if [entry["pair_id"] for entry in pair_rows] != list(range(N)):
        raise QN1Error("retained per-pair denominator is not exactly 600 ordered rows")
    pair_rows_fact = qbt1.atomic_json(output / "PAIR_ROWS.json", pair_rows)
    components = aggregate(pair_rows, archive_bytes=len(archive), class_names=br2.CLASS_NAMES)
    verdict = "SUB-0.12-CANDIDATE" if components["S"] < TARGET_SCORE else "DISTORTION-REFUSED"
    falsifier = ticket["prediction"]
    result = {
        "schema": REALIZED_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "pointer_moved": False,
        "verdict": verdict,
        "verdict_scope": f"INSTANCE (exact QBR1 {ticket['winner']['cell_id']} archive, {len(archive)} bytes)",
        "falsifier_met": bool(components["d_seg"] <= FALSIFIER_D_SEG and components["d_pose"] <= FALSIFIER_D_POSE),
        "prediction": falsifier,
        "components": components,
        "n": N,
        "ticket": file_fact(ticket_path),
        "winner": ticket["winner"],
        "object": obj,
        "container_at_fire": live_container,
        "storage_preflight": storage,
        "claim": claim,
        "retained_chunks": chunks,
        "per_pair_rows": pair_rows_fact,
        "all_renders_logits_argmax_pose_and_targets_retained": True,
        "chunk_pairs": CHUNK_PAIRS,
        "elapsed_seconds": time.time() - started,
        "n32_advisory_numbers_are_not_transferred_to_n600": True,
        "run_config": {"argv": list(sys.argv), "cwd": str(Path.cwd().resolve()), "seed": qbz1.SEED, "device": "cpu"},
        "boundaries": ticket["boundaries"],
        "contest_eval_invocations": 0,
        "modal_invocations": 0,
    }
    qbt1.atomic_json(output / "REALIZED_RESULT.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    ticket = sub.add_parser("ticket", help="write the sealed n600 realization fire order")
    ticket.add_argument("--adjudication", type=Path, default=ADJUDICATION_RESULT)
    ticket.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    ticket.add_argument("--scorer-claim-id", default=CLAIM_PLACEHOLDER)
    ticket.add_argument("--realization-output", type=Path, default=REALIZATION_ROOT)

    dry = sub.add_parser("dry-run", help="scorer-free plumbing proof over an existing milestone")
    dry.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    dry.add_argument("--cell-root", type=Path, default=DRY_RUN_CELL_ROOT)
    dry.add_argument("--milestone", type=Path, default=None)

    run = sub.add_parser("realize", help="MAIN-only BR2-protocol n600 realization of a sealed ticket")
    run.add_argument("--ticket", type=Path, required=True)
    run.add_argument("--output", type=Path, default=REALIZATION_ROOT)
    run.add_argument("--resume-from", type=Path, required=True)
    run.add_argument("--scorer-claim-id", required=True)
    run.add_argument("--launch-authorized", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "ticket":
        payload: Any = ticket_action(
            adjudication_path=args.adjudication,
            output=args.output,
            claim_id=args.scorer_claim_id,
            realization_output=args.realization_output,
        )
    elif args.action == "dry-run":
        payload = dry_run(output=args.output, cell_root=args.cell_root, milestone_path=args.milestone)
    else:
        payload = realize(
            ticket_path=args.ticket,
            output=args.output,
            resume_from=args.resume_from,
            claim_id=args.scorer_claim_id,
            launch_authorized=args.launch_authorized,
        )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
