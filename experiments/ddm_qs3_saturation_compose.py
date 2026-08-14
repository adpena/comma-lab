#!/usr/bin/env python3
"""QS3 scorer-free saturation composition and fail-closed T4 post-mortem.

The runner consumes retained QS1/QS2/JS6 artifacts without launching Modal or
SegNet.  It verifies the downloaded QS1 run, measures a nine-pair shared
compensation-codebook bracket, screens the complete 200-row JS6 bank at the
measured QS1 realization calibration, and writes a sealed no-fire receipt when
the T4 GT field required for exact pixel attribution is unavailable.

Mechanism variants are deliberately labelled TOY-BRACKET.  They do not become
receiver candidates or family verdicts until the retained-field post-mortem,
Schur solve, real coder, and unchanged worker gates all pass.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import shutil
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_qs1_frame0_schur_coupled_solve as qs1
from experiments import ddm_qs2_compensation_rate_rung as qs2
from experiments import ddm_qs3_compensation_overlay_runtime as qs3_overlay

OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_qs3_20260813")
QS1_FIELDS: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qs1_20260813/retained_fields/"
    "ddm_qs1_dual_axis_20260813_r2"
)
QS1_DONE: Final = REPO / ".omx/tmp/codex_runs/qs1_field_download.done"
QS1_BASE_FIELD: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_po1_20260813/round1_cp135/"
    "retained/fields/candidate_argmax_n600.npy"
)
QS1_CANDIDATE_FIELD: Final = QS1_FIELDS / "retained/fields/candidate_argmax_n600.npy"
QS1_GT_FIELD: Final = OUTPUT / "retained/inputs/gt_argmax_n600.npy"
QS2_REMOTE_RESULT: Final = (
    qs2.OUTPUT
    / "dispatch/ddm_qs2_dual_axis_20260813_r2/QS2_T4_REMOTE_RESULT.json"
)
JS6_INDEX: Final = (
    qs1.JS6_BANK / "proposal_index.jsonl"
)
EXPECTED_BASE_SHA256: Final = (
    "7648ad42e9f21942f86e81b97cabf46b710af747bba0909f7837ef3891232727"
)
EXPECTED_CANDIDATE_SHA256: Final = (
    "ad1e3dcc0a57c53f0757773a018335924afc26992f398c23ec084eecace7ed20"
)
EXPECTED_GT_SHA256: Final = (
    "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248"
)
PAIR_COUNT: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
DENOMINATOR_PIXELS: Final = PAIR_COUNT * HEIGHT * WIDTH
MEASURED_CHANGED_PIXELS: Final = 189
MEASURED_NET_FLIPS: Final = 32
CALIBRATED_REALIZATION_EFFICIENCY: Final = (
    MEASURED_NET_FLIPS / MEASURED_CHANGED_PIXELS
)
BREAKEVEN_FLIPS_PER_BYTE: Final = 0.785
RATE_S_PER_BYTE: Final = 25.0 / 37_545_489
AXIS: Final = "[macOS-CPU scorer-free retained-payload analysis]"
QS2_REFERENCE_COMMIT: Final = "d77fb69efc390bf9cbb41dab90d10400300180e5"
QS2_RATE_RUNNER_SHA256: Final = (
    "8654e6d325212acf9a2260a3f3ef73494231f68cae5003295eceaa37537813f1"
)
QS2_OVERLAY_SHA256: Final = (
    "7e5d905d42cd0ec65851d5df5f762ce8adac65783781015ad48585a9fc91231f"
)


class QS3Error(RuntimeError):
    """A retained input, deterministic coder, or fail-closed gate differed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    if not path.is_file():
        raise QS3Error(f"required file is absent: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path) if hash_file else None,
    }


def retain_json(path: Path, value: Any) -> dict[str, Any]:
    qs1.atomic_json(path, value)
    return file_record(path)


def _remote_to_local(remote: str) -> Path | None:
    marker = "/ddm_qs1_dual_axis_20260813_r2/"
    if marker not in remote:
        return None
    return QS1_FIELDS / remote.split(marker, 1)[1]


def _records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            yield value
        for child in value.values():
            yield from _records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _records(child)


def _batch_census(root: Path) -> dict[str, Any]:
    receipts = sorted(root.glob("batch_*/BATCH_RESULT.json"))
    ranges = []
    for path in receipts:
        row = json.loads(path.read_text())
        if row.get("complete") is not True:
            raise QS3Error(f"incomplete retained batch: {path}")
        ranges.append((int(row["pair_start"]), int(row["pair_end"])))
    covered = [pair for start, end in sorted(ranges) for pair in range(start, end)]
    return {
        "root": str(root.resolve()),
        "batch_receipts": len(receipts),
        "covered_pairs": len(covered),
        "exact_n600_partition": covered == list(range(PAIR_COUNT)),
    }


def verify_download(output: Path) -> dict[str, Any]:
    final_path = QS1_FIELDS / "FINAL_RESULT.json"
    final = json.loads(final_path.read_text())
    done_text = QS1_DONE.read_text().strip() if QS1_DONE.is_file() else None
    done_valid = bool(done_text and done_text.split()[0] == "rc=0")
    batch_roots = {
        "seg_candidate": QS1_FIELDS / "retained/scorer/candidate/batches",
        "pose_candidate_first": QS1_FIELDS / "retained/pose/candidate_first/batches",
        "pose_candidate_repeat": QS1_FIELDS / "retained/pose/candidate_repeat/batches",
        "pose_gt": QS1_FIELDS / "retained/pose/gt/batches",
    }
    censuses = {name: _batch_census(path) for name, path in batch_roots.items()}
    in_run_records: dict[str, dict[str, Any]] = {}
    unresolved_external: dict[str, dict[str, Any]] = {}
    json_paths = sorted(QS1_FIELDS.rglob("*.json"))
    for json_path in json_paths:
        try:
            value = json.loads(json_path.read_text())
        except json.JSONDecodeError as error:
            raise QS3Error(f"invalid retained JSON: {json_path}") from error
        for record in _records(value):
            remote = str(record["path"])
            local = _remote_to_local(remote)
            if local is None:
                if remote.startswith("/ddm_js1b_retained/"):
                    unresolved_external[remote] = record
                continue
            in_run_records[str(local)] = record
    absent = []
    size_mismatch = []
    for local_text, record in sorted(in_run_records.items()):
        local = Path(local_text)
        if not local.is_file():
            absent.append(local_text)
        elif local.stat().st_size != int(record["bytes"]):
            size_mismatch.append(
                {"path": local_text, "expected": record["bytes"], "actual": local.stat().st_size}
            )
    base = file_record(QS1_BASE_FIELD)
    candidate = file_record(QS1_CANDIDATE_FIELD)
    gt = file_record(QS1_GT_FIELD) if QS1_GT_FIELD.is_file() else None
    pins = {
        "base_matches": base["sha256"] == EXPECTED_BASE_SHA256,
        "candidate_matches": candidate["sha256"] == EXPECTED_CANDIDATE_SHA256,
        "gt_matches": bool(gt and gt["sha256"] == EXPECTED_GT_SHA256),
    }
    download_set_complete = (
        not absent
        and not size_mismatch
        and all(row["exact_n600_partition"] for row in censuses.values())
    )
    trusted_for_postmortem = (
        done_valid and download_set_complete and all(pins.values())
    )
    blockers = []
    if not done_valid:
        blockers.append("the required .done receipt is absent or does not report rc=0")
    if not pins["gt_matches"]:
        blockers.append("the exact matched T4 GT argmax field is absent")
    if not download_set_complete:
        blockers.append("the downloaded in-run payload census is incomplete")
    result = {
        "schema": "ddm_qs3_qs1_download_verification.v1",
        "qs1_final": file_record(final_path),
        "run_execution_status": final.get("execution_status"),
        "done_receipt": {
            "path": str(QS1_DONE.resolve()),
            "present": QS1_DONE.is_file(),
            "valid_rc0": done_valid,
            "content": done_text,
            "file": file_record(QS1_DONE) if QS1_DONE.is_file() else None,
        },
        "json_documents_checked": len(json_paths),
        "referenced_in_run_files": len(in_run_records),
        "referenced_in_run_absent": absent,
        "referenced_in_run_size_mismatch": size_mismatch,
        "unresolved_external_records": unresolved_external,
        "batch_censuses": censuses,
        "fields": {"base": base, "candidate": candidate, "gt": gt},
        "field_pins": pins,
        "download_set_complete_except_external_priors": download_set_complete,
        "trusted_for_postmortem": trusted_for_postmortem,
        "blocker": None if trusted_for_postmortem else (
            "; ".join(blockers)
            + "; the 157-pixel authority set cannot be identified from aggregate counts"
        ),
        "qs3_created_done_receipt": False,
        "score_claim": False,
    }
    retain_json(output / "QS1_FIELD_DOWNLOAD_VERIFICATION.json", result)
    return result


def _qs1_selected_rows() -> list[dict[str, Any]]:
    screen = json.loads(qs2.QS1_SCREEN.read_text())
    rows = []
    for proposal_id in screen["selected_proposal_ids"]:
        path = qs2.QS1_STORE / "retained/proposals" / proposal_id / "RESULT.json"
        row = json.loads(path.read_text())
        row["result_record"] = file_record(path)
        rows.append(row)
    return sorted(rows, key=lambda row: int(row["pair"]))


def changed_field_census(output: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    base = np.load(QS1_BASE_FIELD, mmap_mode="r", allow_pickle=False)
    candidate = np.load(QS1_CANDIDATE_FIELD, mmap_mode="r", allow_pickle=False)
    if base.shape != (PAIR_COUNT, HEIGHT, WIDTH) or candidate.shape != base.shape:
        raise QS3Error("retained argmax field geometry differs")
    changed_per_pair = np.count_nonzero(base != candidate, axis=(1, 2))
    changed_pairs = np.flatnonzero(changed_per_pair)
    expected_pairs = np.asarray([int(row["pair"]) for row in rows], dtype=np.int64)
    if not np.array_equal(changed_pairs, expected_pairs):
        raise QS3Error(f"changed pair set differs: {changed_pairs.tolist()}")
    table = []
    for row in rows:
        pair = int(row["pair"])
        proposal_id = str(row["proposal_id"])
        proposal = json.loads((qs1.JS6_BANK / "proposals" / proposal_id / "proposal.json").read_text())
        changed = base[pair] != candidate[pair]
        source = int(proposal["source_class_id"])
        target = int(proposal["target_class_id"])
        target_component = np.load(
            qs1.JS6_BANK / "proposals" / proposal_id / "target_component.bool.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        table.append(
            {
                "pair": pair,
                "proposal_id": proposal_id,
                "directed_edge": proposal["directed_edge"],
                "changed_pixels": int(np.count_nonzero(changed)),
                "directed_source_to_target": int(
                    np.count_nonzero(changed & (base[pair] == source) & (candidate[pair] == target))
                ),
                "reverse_target_to_source": int(
                    np.count_nonzero(changed & (base[pair] == target) & (candidate[pair] == source))
                ),
                "off_directed_edge": int(
                    np.count_nonzero(
                        changed
                        & ~((base[pair] == source) & (candidate[pair] == target))
                        & ~((base[pair] == target) & (candidate[pair] == source))
                    )
                ),
                "changed_inside_target_component": int(
                    np.count_nonzero(changed & target_component)
                ),
            }
        )
    total = int(changed_per_pair.sum())
    if total != MEASURED_CHANGED_PIXELS:
        raise QS3Error(f"retained changed-pixel census differs: {total}")
    result = {
        "schema": "ddm_qs3_changed_field_census.v1",
        "axis": "[contest-CUDA T4 retained candidate/base argmax fields, no GT attribution] COMPONENT-ONLY",
        "selection_mode": "all 600 pairs; every base/candidate argmax difference",
        "denominator_pixels": DENOMINATOR_PIXELS,
        "changed_pixels": total,
        "measured_net_beneficial_flips_from_worker_aggregate": MEASURED_NET_FLIPS,
        "arithmetic_changed_minus_net": total - MEASURED_NET_FLIPS,
        "warning": (
            "157 is aggregate arithmetic, not an identifiable pixel set. Beneficial, harmful, "
            "and wrong-to-different-wrong pixels require the exact matched T4 GT field."
        ),
        "mechanism_taxonomy": {
            "sub_quantum_amplitude": None,
            "aa_resize_washout": None,
            "tie_margin_failure": None,
            "other": None,
            "status": "BLOCKED_MISSING_MATCHED_T4_GT_FIELD",
            "denominator_requested": total - MEASURED_NET_FLIPS,
        },
        "per_pair": table,
        "score_claim": False,
    }
    retain_json(output / "POSTMORTEM_BLOCKED_CENSUS.json", result)
    return result


def _all_qs1_rows() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((qs2.QS1_STORE / "retained/proposals").glob("*/RESULT.json")):
        row = json.loads(path.read_text())
        row["result_record"] = file_record(path)
        rows.append(row)
    return rows


def unique_pair_calibration_rows() -> list[dict[str, Any]]:
    best: dict[int, dict[str, Any]] = {}
    for row in _all_qs1_rows():
        pair = int(row["pair"])
        incumbent = best.get(pair)
        if incumbent is None or float(row["screen"]["screen_margin_s"]) > float(
            incumbent["screen"]["screen_margin_s"]
        ):
            best[pair] = row
    return [best[pair] for pair in sorted(best)]


def codebook_race(output: Path) -> dict[str, Any]:
    if sha256_file(Path(qs2.__file__).resolve()) != QS2_RATE_RUNNER_SHA256:
        raise QS3Error("QS2 reference rate runner differs from commit d77fb69efc")
    if sha256_file(qs2.RUNTIME_OVERLAY_SOURCE) != QS2_OVERLAY_SHA256:
        raise QS3Error("QS2 reference overlay differs from commit d77fb69efc")
    rows = unique_pair_calibration_rows()
    pairs, exact = qs2.exact_deltas(rows)
    sources = qs2._candidate_rate_sources()
    candidates = []
    winners = {}
    for step in (1, 2):
        label = f"nine_pair_deadzone_step_{step}"
        deltas = qs2.deadzone_quantize(exact, step)
        step_rows = [
            build_scale_rate_candidate(
                output=output,
                label=label,
                pair_indices=pairs,
                deltas=deltas,
                carrier_quality=quality,
                sources=sources,
            )
            for quality in range(12)
        ]
        winner = min(step_rows, key=lambda row: (row["archive"]["bytes"], row["carrier_quality"]))
        candidates.extend(step_rows)
        winners[label] = winner
    exact_winner = winners["nine_pair_deadzone_step_1"]
    step2_winner = winners["nine_pair_deadzone_step_2"]
    result = {
        "schema": "ddm_qs3_shared_codebook_race.v1",
        "axis": "[macOS-CPU exact byte/container measurement]",
        "reference_form": {
            "commit": QS2_REFERENCE_COMMIT,
            "rate_runner_sha256": QS2_RATE_RUNNER_SHA256,
            "overlay_sha256": QS2_OVERLAY_SHA256,
            "qs1_compile_store": str(qs2.QS1_STORE.resolve()),
            "qs2_store": str(qs2.OUTPUT.resolve()),
            "js6_bank": str(qs1.JS6_BANK.resolve()),
        },
        "verdict_scope": (
            "FORMULATION: Q3C1 four-bit shared compensation overlay on nine unique retained "
            "QS1 Schur rows; semantic suffix remains the six-pair QS1 object and no contest "
            "runtime consumes Q3C1, so this is a TOY-BRACKET coding calibration, not a "
            "receiver-closed composed candidate"
        ),
        "selection_mode": "minimum exact retained archive bytes over q0..q11 for each rung",
        "candidate_denominator": len(candidates),
        "pair_denominator": len(rows),
        "pairs": pairs.astype(int).tolist(),
        "winners": winners,
        "exact_total_archive_delta_bytes_per_calibration_pair": (
            exact_winner["archive_delta_bytes_vs_cp135"] / len(rows)
        ),
        "beats_qs2_5_67_total_delta_bytes_per_pair": (
            exact_winner["archive_delta_bytes_vs_cp135"] / len(rows) < 5.67
        ),
        "step_2_active_pair_denominator": len(step2_winner["pair_indices"]),
        "step_2_total_delta_bytes_per_active_pair": step2_winner[
            "bytes_per_active_pair"
        ],
        "step_2_beats_qs2_5_67_total_delta_bytes_per_pair": (
            step2_winner["bytes_per_active_pair"] < 5.67
        ),
        "deadzone_step_2_reraced": True,
        "all_payloads_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    retain_json(output / "SHARED_CODEBOOK_RACE.json", result)
    retain_json(output / "checkpoints/stage_30_codebook_race.json", result)
    return result


def consume_qs2_calibration(output: Path) -> dict[str, Any]:
    if not QS2_REMOTE_RESULT.is_file():
        result = {
            "schema": "ddm_qs3_qs2_calibration.v1",
            "status": "IN_FLIGHT",
            "expected_path": str(QS2_REMOTE_RESULT.resolve()),
            "score_claim": False,
        }
        retain_json(output / "QS2_CALIBRATION.json", result)
        return result
    wrapper = json.loads(QS2_REMOTE_RESULT.read_text())
    raw = wrapper["artifacts"]["QS1_T4_REMOTE_RESULT.json"]
    payload = ast.literal_eval(raw)
    worker = json.loads(payload.decode())
    field = worker["field_measurement"]
    pose = worker["pose_measurement"]
    base_dpose = 6.885642960696714e-6
    candidate_dpose = float(pose["d_pose_candidate_first"])
    seg_delta_s = 100.0 * (
        int(field["candidate_flips_vs_gt"]) - int(field["base_flips_vs_gt"])
    ) / DENOMINATOR_PIXELS
    pose_delta_s = float((10.0 * candidate_dpose) ** 0.5 - (10.0 * base_dpose) ** 0.5)
    delta_bytes = int(worker["candidate_archive"]["bytes"]) - qs2.CP135_BYTES
    rate_delta_s = delta_bytes * RATE_S_PER_BYTE
    total_delta_s = seg_delta_s + pose_delta_s + rate_delta_s
    result = {
        "schema": "ddm_qs3_qs2_calibration.v1",
        "status": "CONSUMED",
        "source": file_record(QS2_REMOTE_RESULT),
        "axis": wrapper["axis"],
        "base_flips": int(field["base_flips_vs_gt"]),
        "candidate_flips": int(field["candidate_flips_vs_gt"]),
        "net_beneficial_flips": -int(field["candidate_minus_base_flips"]),
        "base_dpose": base_dpose,
        "candidate_dpose": candidate_dpose,
        "archive_delta_bytes": delta_bytes,
        "seg_delta_s": seg_delta_s,
        "pose_delta_s": pose_delta_s,
        "rate_delta_s": rate_delta_s,
        "recomputed_total_delta_s": total_delta_s,
        "band_verdict": "SUB_BAND" if abs(total_delta_s) < 1e-5 else "SUPER_BAND",
        "worker_payload_repeat_deterministic": (
            pose["d_pose_candidate_first"] == pose["d_pose_candidate_repeat"]
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }
    retain_json(output / "QS2_CALIBRATION.json", result)
    retain_json(output / "checkpoints/stage_25_qs2_calibration.json", result)
    return result


def build_scale_rate_candidate(
    *,
    output: Path,
    label: str,
    pair_indices: np.ndarray,
    deltas: np.ndarray,
    carrier_quality: int,
    sources: tuple[bytes, bytes, bytes, bytes],
) -> dict[str, Any]:
    candidate_root = output / "retained/rate_race" / label / f"q{carrier_quality:02d}"
    result_path = candidate_root / "RESULT.json"
    active = np.any(deltas != 0, axis=1)
    active_pairs = np.asarray(pair_indices[active], dtype=np.int16)
    active_deltas = np.asarray(deltas[active], dtype=np.int32)
    overlay = qs3_overlay.encode_compensation_overlay(active_pairs, active_deltas)
    decoded_pairs, decoded_deltas = qs3_overlay.decode_compensation_overlay(overlay)
    if not np.array_equal(decoded_pairs, active_pairs) or not np.array_equal(
        decoded_deltas, active_deltas
    ):
        raise QS3Error("Q3C1 shared overlay parse-back differs")
    stream_a, stream_b, base_carrier, suffix = sources
    carrier_source = base_carrier + overlay
    stream_c = qs2._brotli_compress(carrier_source, carrier_quality)
    models = qs2.SPLIT_HEADER.pack(len(stream_a), len(stream_b), len(stream_c))
    models += stream_a + stream_b + stream_c
    member = models + suffix
    archive = qs2.deterministic_zip(member)
    archive_repeat = qs2.deterministic_zip(member)
    if archive_repeat != archive:
        raise QS3Error("independent deterministic archive repeat differs")
    records = {
        "overlay": qs1.retain_bytes(candidate_root / "compensation.q3c1", overlay),
        "carrier_source": qs1.retain_bytes(
            candidate_root / "carrier_selector_plus_overlay.raw", carrier_source
        ),
        "carrier_stream": qs1.retain_bytes(
            candidate_root / f"carrier_selector_plus_overlay.q{carrier_quality:02d}.br",
            stream_c,
        ),
        "split_models": qs1.retain_bytes(candidate_root / "split_models.bin", models),
        "member": qs1.retain_bytes(candidate_root / "p", member),
        "archive": qs1.retain_bytes(candidate_root / "archive.zip", archive),
        "archive_repeat": qs1.retain_bytes(
            candidate_root / "archive.repeat.zip", archive_repeat
        ),
    }
    delta_bytes = records["archive"]["bytes"] - qs2.CP135_BYTES
    result = {
        "schema": "ddm_qs3_scale_rate_candidate.v1",
        "label": label,
        "carrier_quality": carrier_quality,
        "pair_indices": active_pairs.astype(int).tolist(),
        "deltas": active_deltas.astype(int).tolist(),
        **records,
        "archive_delta_bytes_vs_cp135": delta_bytes,
        "bytes_per_active_pair": delta_bytes / len(active_pairs),
        "rate_delta_s": delta_bytes * RATE_S_PER_BYTE,
        "receiver_closed": False,
        "verdict_scope": "TOY-BRACKET Q3C1 scale code; no contest runtime consumer",
        "all_payloads_retained": True,
        "archive_repeat_byte_identical": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    retain_json(result_path, result)
    return result


def load_js6_rows() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in JS6_INDEX.read_text().splitlines() if line]
    if len(rows) != 200 or len({row["proposal_id"] for row in rows}) != 200:
        raise QS3Error("JS6 bank is not the sealed 200-row census")
    return rows


def screening_row(
    row: dict[str, Any], *, bytes_per_pair: float, pose_s_per_cell: float
) -> dict[str, Any]:
    target_mass = int(row["receiver_surface"]["exact_field_target_edge_mass_on_support"])
    predicted_flips = target_mass * CALIBRATED_REALIZATION_EFFICIENCY
    predicted_flips_per_byte = predicted_flips / bytes_per_pair
    pose_s = int(row["token_site_count"]) * pose_s_per_cell
    rate_s = bytes_per_pair * RATE_S_PER_BYTE
    adjusted_bar = BREAKEVEN_FLIPS_PER_BYTE * (1.0 + pose_s / rate_s)
    return {
        "proposal_id": row["proposal_id"],
        "pair": int(row["pair"]),
        "directed_edge": row["directed_edge"],
        "target_mass_upper_bound": target_mass,
        "token_site_count": int(row["token_site_count"]),
        "calibrated_efficiency": CALIBRATED_REALIZATION_EFFICIENCY,
        "predicted_flips": predicted_flips,
        "measured_codebook_bytes_per_pair": bytes_per_pair,
        "predicted_flips_per_byte": predicted_flips_per_byte,
        "projected_schur_pose_s": pose_s,
        "rate_s": rate_s,
        "pose_adjusted_admission_bar_flips_per_byte": adjusted_bar,
        "passes_toy_bracket": predicted_flips_per_byte > adjusted_bar,
        "admission_status": "QUEUE_FOR_EXACT_SCHUR_AND_MATCHED_T4_GT_POSTMORTEM",
        "verdict_scope": "TOY-BRACKET projection; not an admitted receiver candidate",
    }


def survival_and_bank_screen(output: Path, codebook: dict[str, Any]) -> dict[str, Any]:
    rows = load_js6_rows()
    exact_winner = codebook["winners"]["nine_pair_deadzone_step_1"]
    bytes_per_pair = float(exact_winner["archive_delta_bytes_vs_cp135"]) / float(
        codebook["pair_denominator"]
    )
    calibration_rows = _all_qs1_rows()
    pose_per_cell = np.asarray(
        [
            float(row["pose"]["conservative_residual_pose_bound_s_at_603"])
            / int(json.loads((qs1.JS6_BANK / "proposals" / row["proposal_id"] / "proposal.json").read_text())["token_site_count"])
            for row in calibration_rows
        ],
        dtype=np.float64,
    )
    pose_s_per_cell = float(np.median(pose_per_cell))
    screen = [
        screening_row(row, bytes_per_pair=bytes_per_pair, pose_s_per_cell=pose_s_per_cell)
        for row in rows
    ]
    screen.sort(key=lambda row: (-row["predicted_flips_per_byte"], row["proposal_id"]))
    screen_payload = b"".join(
        (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode() for row in screen
    )
    screen_record = qs1.retain_bytes(output / "retained/full_bank_screen.jsonl", screen_payload)

    variants = []
    for row in rows:
        proposal_root = qs1.JS6_BANK / "proposals" / row["proposal_id"]
        target = np.load(proposal_root / "target_component.bool.npy", mmap_mode="r", allow_pickle=False)
        strong = np.load(proposal_root / "strong_support.bool.npy", mmap_mode="r", allow_pickle=False)
        component = int(np.count_nonzero(target))
        above = int(np.count_nonzero(target & strong))
        below = component - above
        for variant, status in (
            ("amplitude_above_js5_quantum", "GENERATED_RECIPE"),
            ("d_aware_precompensation", "GENERATED_RECIPE"),
            ("margin_targeted_placement", "BLOCKED_MISSING_MATCHED_T4_BASE_LOGITS"),
        ):
            variants.append(
                {
                    "schema": "ddm_qs3_survival_recipe.v1",
                    "proposal_id": row["proposal_id"],
                    "pair": int(row["pair"]),
                    "variant": variant,
                    "status": status,
                    "component_pixels": component,
                    "component_pixels_above_js5_quantum": above,
                    "component_pixels_below_js5_quantum": below,
                    "quantum_coverage": above / component,
                    "efficiency_curve": [
                        {
                            "efficiency": efficiency,
                            "projected_flips": int(row["receiver_surface"]["exact_field_target_edge_mass_on_support"]) * efficiency,
                        }
                        for efficiency in (
                            CALIBRATED_REALIZATION_EFFICIENCY,
                            0.25,
                            1.0 / 3.0,
                            0.5,
                        )
                    ],
                    "receiver_payload_generated": False,
                    "verdict_scope": "TOY-BRACKET recipe only",
                }
            )
    recipe_payload = b"".join(
        (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode() for row in variants
    )
    recipe_record = qs1.retain_bytes(
        output / "retained/survival_engineered_recipes.jsonl", recipe_payload
    )
    result = {
        "schema": "ddm_qs3_full_bank_screen.v1",
        "axis": AXIS,
        "selection_mode": "complete 200-row JS6 bank; no sampling or prefix",
        "bank_denominator": len(rows),
        "calibration_pair_denominator": codebook["pair_denominator"],
        "calibration_efficiency": CALIBRATED_REALIZATION_EFFICIENCY,
        "measured_codebook_bytes_per_pair": bytes_per_pair,
        "projected_schur_pose_s_per_semantic_cell": pose_s_per_cell,
        "toy_bracket_pass_count": sum(row["passes_toy_bracket"] for row in screen),
        "screen_rows": screen_record,
        "survival_recipe_denominator": len(variants),
        "survival_recipes": recipe_record,
        "gca1_effect": (
            "GCA1 permits graph distance/heat only as a prioritizer; no finite edit-radius theorem "
            "was imported into admission."
        ),
        "downstream_status": "BLOCKED_BEFORE_EXACT_SCHUR_ADMISSION",
        "score_claim": False,
        "promotion_eligible": False,
    }
    retain_json(output / "FULL_BANK_SCREEN.json", result)
    retain_json(output / "checkpoints/stage_40_full_bank_screen.json", result)
    return result


def storage_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output).free
    required = 8 * 1024**3
    result = {
        "schema": "ddm_qs3_storage_preflight.v1",
        "tier": str(output.resolve()),
        "free_bytes": free,
        "required_free_bytes": required,
        "passed": free >= required,
        "cleanup_policy": "certify-or-block; no generated payload deleted or moved",
    }
    retain_json(output / "STORAGE_PREFLIGHT.json", result)
    if not result["passed"]:
        raise QS3Error("SSD storage preflight failed")
    return result


def finalize(
    output: Path,
    *,
    verification: dict[str, Any],
    census: dict[str, Any],
    qs2_calibration: dict[str, Any],
    codebook: dict[str, Any],
    bank: dict[str, Any],
) -> dict[str, Any]:
    blocker = {
        "schema": "ddm_qs3_sealed_no_fire_order.v1",
        "sealed": True,
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN sole scorer-lane router",
        "consumer_store": str(output.resolve()),
        "fire_trigger": (
            "the exact T4 GT field with sha256 91d3ff11... is present locally, the original "
            ".done receipt exists, the resumed QS3 post-mortem assigns all authority pixels, "
            "and one receiver-closed candidate clears net realized delta S < 0 on the matched "
            "34970-flip and 6.885642960696714e-6 d_pose base"
        ),
        "recovery_command_argv": [
            ".venv/bin/modal",
            "volume",
            "get",
            "comma-ddm-js1b-argmax-retained",
            "ddm_js1b_20260813b/retained/fields/gt_argmax_n600.npy",
            str(QS1_GT_FIELD),
            "--force",
        ],
        "resume_command_argv": [
            ".venv/bin/python",
            "experiments/ddm_qs3_saturation_compose.py",
            "--resume-from",
            str(output.resolve()),
        ],
        "dual_axis_fire_order_status": "NOT_SEALED_UNTIL_PREENCODE_ADMISSION_PASSES",
        "canonical_evaluate_follow_on": "NOT_NAMED_UNTIL_WORKER_SUPER_BAND_RESULT_EXISTS",
        "modal_fired": False,
        "score_claim": False,
    }
    no_fire_record = retain_json(output / "SEALED_NO_FIRE_ORDER.json", blocker)
    result = {
        "schema": "ddm_qs3_final_result.v1",
        "axis": AXIS,
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "download_verification": verification,
        "postmortem": census,
        "qs2_calibration": qs2_calibration,
        "shared_codebook": codebook,
        "full_bank_screen": bank,
        "waterfilled_compile": None,
        "waterfill_status": "BLOCKED_MISSING_MATCHED_T4_GT_FIELD",
        "sealed_no_fire_order": no_fire_record,
        "segnet_rerun": False,
        "modal_fired": False,
        "all_materialized_payloads_retained": True,
        "pointer_moved": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    retain_json(output / "FINAL_RESULT.json", result)
    retain_json(output / "checkpoints/stage_90_final.json", result)
    return result


def run(output: Path = OUTPUT) -> dict[str, Any]:
    if output.resolve() != OUTPUT.resolve():
        raise QS3Error(f"output must be the governed SSD store: {OUTPUT}")
    storage_preflight(output)
    verification = verify_download(output)
    rows = _qs1_selected_rows()
    census = changed_field_census(output, rows)
    qs2_calibration = consume_qs2_calibration(output)
    codebook = codebook_race(output)
    bank = survival_and_bank_screen(output, codebook)
    return finalize(
        output,
        verification=verification,
        census=census,
        qs2_calibration=qs2_calibration,
        codebook=codebook,
        bank=bank,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume-from", type=Path, default=OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume_from.resolve() != args.output.resolve():
        raise QS3Error("--resume-from must equal --output")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "RUN.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise QS3Error("another QS3 process holds the governed run lock") from error
        result = run(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
