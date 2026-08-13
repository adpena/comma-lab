#!/usr/bin/env python3
"""Compose the five VD1-validated EC1 events through JO1's HP3/RC64 path.

This runner is scorer-free.  It retains both independent candidate archives,
the adapted-runtime token decode, and one exact diff receipt per event on the
primary SSD tier.  Every expensive stage is resumable from retained receipts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_jo1_joint_probability_object as jo1
from experiments import ddm_vd1_batch_event_validator_worker as vd1

OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_cp5v_20260812")
CENSUS_FINAL: Final = Path("/Volumes/VertigoDataTier/pact/ddm_vd1_20260812/main_harvest/results/FINAL_RESULT.json")
CENSUS_EVENTS: Final = Path("/Volumes/VertigoDataTier/pact/ddm_vd1_20260812/main_harvest/results/EVENT_RESULTS.jsonl")
CENSUS_FINAL_SHA256: Final = "6c53628184f55722f87fcb7e3dadc8b6c9a70025a804e00cfcbecb6674004973"
CENSUS_EVENTS_SHA256: Final = "a97400d32878318d8eb657a36e62f523e4db48e402b292c09e611d2104b500b3"
JO1_COMMIT: Final = "3bc2cb557f"
JO1_SOURCE: Final = REPO / "experiments/ddm_jo1_joint_probability_object.py"
BASE_ARCHIVE: Final = jo1.CP135_RUNTIME / "archive.zip"
CANDIDATE_NAME: Final = "validated_five"
EVENT_IDS: Final = (
    "ec1_0164_3a4e239de5b9",
    "ec1_0168_818a3c77af51",
    "ec1_0004_3bc2b69c706c",
    "ec1_0104_f4e219067530",
    "ec1_0003_fcb5ca3a4453",
)
AFFECTED_PAIRS: Final = (7, 18, 53, 73, 76, 96)
EXPECTED_FLIP_GAIN: Final = 6
EXPECTED_SEG_SCORE_GAIN: Final = 5.086263020833333e-06
MAX_DELTA_BYTES: Final = 3
MIN_FREE_BYTES: Final = 8 * 1024**3
AXIS: Final = "[macOS-CPU scorer-free direct-token HP3/RC64 n600 reclose]"


class CP5VError(RuntimeError):
    """A CP5V custody, receiver, census, or falsifier invariant failed."""


def save_state(output: Path, active_stage: str, completed_stages: list[str]) -> None:
    jo1.atomic_json(
        output / "state.json",
        {
            "schema": "ddm_cp5v_state.v1",
            "status": "COMPLETE" if active_stage == "complete" else "RUNNING",
            "active_stage": active_stage,
            "completed_stages": completed_stages,
            "resumable": True,
            "scorer_run": False,
            "score_claim": False,
        },
    )


def _require_record(record: dict[str, Any]) -> None:
    path = Path(record["path"])
    if not path.is_file() or jo1.file_record(path) != record:
        raise CP5VError(f"retained artifact failed custody: {path}")


def load_census_rows() -> list[dict[str, Any]]:
    wanted = set(EVENT_IDS)
    selected: dict[str, dict[str, Any]] = {}
    with CENSUS_EVENTS.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            proposal_id = str(row.get("proposal_id", ""))
            if proposal_id in wanted:
                if proposal_id in selected:
                    raise CP5VError(f"duplicate census row: {proposal_id}")
                selected[proposal_id] = row
    if set(selected) != wanted:
        raise CP5VError(f"census event set differs: found={sorted(selected)}")
    rows = [selected[proposal_id] for proposal_id in EVENT_IDS]
    for row in rows:
        if (
            row.get("schema") != "ddm_vd1_singleton_result.v1"
            or row.get("n600_denominator_exact") is not True
            or row.get("downstream_selection_eligible") is not True
            or int(row.get("site_count", -1)) != 1
            or int(row.get("delta_flips_candidate_minus_base", 0)) >= 0
        ):
            raise CP5VError(f"event is not n600 validated and eligible: {row.get('proposal_id')}")
    flip_gain = sum(int(row["net_flip_gain_base_minus_candidate"]) for row in rows)
    if flip_gain != EXPECTED_FLIP_GAIN:
        raise CP5VError(f"validated flip gain differs: {flip_gain}")
    return rows


def preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    if not output.resolve().is_relative_to(Path("/Volumes/VertigoDataTier/pact")):
        raise CP5VError("CP5V output must remain on the primary SSD tier")
    usage = shutil.disk_usage(output)
    if usage.free < MIN_FREE_BYTES:
        raise CP5VError("SSD storage preflight failed closed")
    jo1.preflight(output)
    jo1.require(BASE_ARCHIVE, size=jo1.BASE_BYTES, digest=jo1.BASE_SHA256)
    jo1.require(CENSUS_FINAL, digest=CENSUS_FINAL_SHA256)
    jo1.require(CENSUS_EVENTS, digest=CENSUS_EVENTS_SHA256)
    landed_source = subprocess.run(
        ["git", "show", f"{JO1_COMMIT}:experiments/ddm_jo1_joint_probability_object.py"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    if jo1.sha256_bytes(landed_source) != jo1.sha256_file(JO1_SOURCE):
        raise CP5VError("working JO1 machinery differs from the pinned landed commit")
    rows = load_census_rows()
    result = {
        "schema": "ddm_cp5v_preflight.v1",
        "axis": AXIS,
        "git_head": jo1.git_head(),
        "jo1_landed_commit": JO1_COMMIT,
        "free_bytes": usage.free,
        "required_free_bytes": MIN_FREE_BYTES,
        "base_archive": jo1.file_record(BASE_ARCHIVE),
        "jo1_probability_object_base_archive": jo1.file_record(jo1.BASE_ARCHIVE),
        "jo1_source": jo1.file_record(JO1_SOURCE),
        "base_spatial_tokens": jo1.file_record(jo1.BASE_SPATIAL),
        "census_final": jo1.file_record(CENSUS_FINAL),
        "census_events": jo1.file_record(CENSUS_EVENTS),
        "proposal_store": str(jo1.js6.DEFAULT_PROPOSAL_ROOT),
        "event_ids": list(EVENT_IDS),
        "validated_flip_gain": sum(int(row["net_flip_gain_base_minus_candidate"]) for row in rows),
        "scorer_run": False,
        "score_claim": False,
        "all_payloads_retained": True,
    }
    jo1.atomic_json(output / "05_CP5V_PREFLIGHT.json", result)
    return result


def _load_retained_reclose(output: Path, *, repeat: bool) -> dict[str, Any] | None:
    root = jo1.candidate_root(output, CANDIDATE_NAME, repeat=repeat)
    path = root / "50_RECLOSE_RESULT.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    for key in ("archive", "token", "member", "shipped_decoded_spatial_tokens"):
        _require_record(result[key])
    if result.get("receiver_closed") is not True or result.get("score_claim") is not False:
        raise CP5VError(f"retained reclose receipt is not admissible: {path}")
    return result


def _materialize_and_reclose(output: Path, *, repeat: bool) -> dict[str, Any]:
    retained = _load_retained_reclose(output, repeat=repeat)
    if retained is not None:
        return retained
    jo1.materialize_candidate(output, CANDIDATE_NAME, list(EVENT_IDS), repeat=repeat)
    return jo1.reclose_candidate(output, CANDIDATE_NAME, repeat=repeat)


def compose(output: Path) -> dict[str, Any]:
    primary = _materialize_and_reclose(output, repeat=False)
    primary_root = jo1.candidate_root(output, CANDIDATE_NAME)
    archive_path = Path(primary["archive"]["path"])
    runtime = jo1.copy_runtime(primary_root / "adapted_runtime", archive_path.read_bytes())
    repeated = _materialize_and_reclose(output, repeat=True)
    same_archive = primary["archive"]["sha256"] == repeated["archive"]["sha256"]
    same_token = primary["token"]["sha256"] == repeated["token"]["sha256"]
    delta_bytes = int(primary["delta_bytes_vs_cp135"])
    result = {
        "schema": "ddm_cp5v_compose_result.v1",
        "axis": AXIS,
        "event_ids": list(EVENT_IDS),
        "primary": primary["archive"],
        "independent_repeat": repeated["archive"],
        "independent_archive_byte_identical": same_archive,
        "independent_rc64_byte_identical": same_token,
        "delta_bytes_vs_cp135": delta_bytes,
        "runtime": runtime,
        "receiver_closed_by_jo1_shipped_backend": True,
        "scorer_run": False,
        "score_claim": False,
        "all_payloads_retained": True,
    }
    jo1.atomic_json(output / "20_COMPOSE_RESULT.json", result)
    if delta_bytes > MAX_DELTA_BYTES or not same_archive or not same_token:
        jo1.atomic_json(
            output / "FALSIFIER.json",
            {
                **result,
                "status": "FIRED",
                "reason": "archive exceeds +3 B or independent repeat differs",
            },
        )
        raise CP5VError("compose falsifier fired; retained FALSIFIER.json")
    return result


def _atomic_raw_array(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = np.ascontiguousarray(value, dtype=np.uint8)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        stream.write(memoryview(contiguous).cast("B"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return jo1.file_record(path)


def _decode_with_retained_sparse_logits(
    f26: Any,
    parts: Any,
    renderer: Any,
    code_dir: Path,
    device: Any,
    cache_root: Path,
) -> tuple[Any, dict[str, Any]]:
    """Run the canonical reader with crash-resumable selected-logit payloads."""
    import torch

    base_sparse_class = f26._sparse_class(code_dir)
    original_factory = f26._sparse_class

    class RetainedSparseIntegerHPAC(base_sparse_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._cp5v_call_index = 0

        def selected_logits(self, current: Any, context: Any, group: int) -> Any:
            call_index = self._cp5v_call_index
            self._cp5v_call_index += 1
            group_count = len(self.plans)
            frame, expected_group = divmod(call_index, group_count)
            if int(group) != expected_group:
                raise CP5VError(
                    f"canonical sparse call order drifted: frame={frame} "
                    f"expected_group={expected_group} actual_group={group}"
                )
            frame_root = cache_root / f"frame_{frame:04d}"
            logits_path = frame_root / f"group_{group:02d}.float32.npy"
            receipt_path = frame_root / f"group_{group:02d}.json"
            current_sha = jo1.raw_array_sha256(current.detach().cpu().numpy())
            if logits_path.is_file() and receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text())
                if (
                    int(receipt.get("frame", -1)) != frame
                    or int(receipt.get("group", -1)) != group
                    or receipt.get("current_token_sha256") != current_sha
                    or jo1.file_record(logits_path) != receipt.get("selected_logits")
                ):
                    raise CP5VError(f"retained canonical sparse cache failed custody: {receipt_path}")
                values = np.load(logits_path, mmap_mode="r", allow_pickle=False)
                return torch.from_numpy(np.array(values, copy=True)).to(device)
            values = super().selected_logits(current, context, group)
            record = vd1.atomic_npy(
                logits_path,
                values.detach().cpu().numpy().astype(np.float32, copy=False),
            )
            vd1.atomic_json(
                receipt_path,
                {
                    "schema": "ddm_cp5v_canonical_sparse_logit_checkpoint.v1",
                    "frame": frame,
                    "group": int(group),
                    "current_token_sha256": current_sha,
                    "selected_logits": record,
                    "complete": True,
                },
            )
            return values

    f26._sparse_class = lambda supplied_code_dir: (
        RetainedSparseIntegerHPAC
        if Path(supplied_code_dir).resolve() == code_dir.resolve()
        else original_factory(supplied_code_dir)
    )
    try:
        tokens, report = f26.decode_production_tokens(parts, renderer, code_dir, device)
    finally:
        f26._sparse_class = original_factory
    checkpoint_count = len(list(cache_root.glob("frame_*/group_*.json")))
    return tokens, {
        **report,
        "resumable_selected_logit_checkpoints": checkpoint_count,
        "resumable_selected_logit_cache_root": str(cache_root.resolve()),
        "canonical_reader_function_invoked": "runtime.f26_inflate.decode_production_tokens",
    }


def decode_adapted_runtime(output: Path) -> dict[str, Any]:
    retained = output / "retained/adapted_runtime_decode"
    result_path = retained / "TOKEN_DECODE_RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        for key in ("archive", "decoded_tokens", "adapted_runtime_module", "rc64_library"):
            _require_record(result[key])
        if result.get("canonical_reader_match_to_jo1_shipped_backend") is not True:
            raise CP5VError("retained canonical-reader receipt is not receiver-closed")
        return result

    primary_root = jo1.candidate_root(output, CANDIDATE_NAME)
    archive_path = primary_root / "objects/archive.zip"
    runtime_root = primary_root / "adapted_runtime"
    dependency = vd1.ensure_adapted_runtime_brotli(
        runtime_root,
        retained / "runtime_dependencies/brotli",
    )
    f26 = vd1._load_adapted_f26(runtime_root)
    parts = f26.read_residual_archive(archive_path)
    renderer = f26._load_renderer(runtime_root / "cpr1")
    receiver_build = retained / "receiver_build"
    receiver_build.mkdir(parents=True, exist_ok=True)
    rc64_library = vd1.compile_rc64(runtime_root, receiver_build)
    import torch

    prior_library = os.environ.get("CPR1_RC64_LIBRARY")
    os.environ["CPR1_RC64_LIBRARY"] = str(rc64_library)
    try:
        tokens, token_report = _decode_with_retained_sparse_logits(
            f26,
            parts,
            renderer,
            runtime_root / "cpr1",
            torch.device("cpu"),
            retained / "canonical_reader_cache/selected_logits",
        )
    finally:
        if prior_library is None:
            os.environ.pop("CPR1_RC64_LIBRARY", None)
        else:
            os.environ["CPR1_RC64_LIBRARY"] = prior_library
    array = tokens.detach().cpu().numpy()
    if tuple(array.shape) != (jo1.FRAMES, jo1.HEIGHT, jo1.WIDTH) or array.dtype != np.uint8:
        raise CP5VError(f"adapted runtime decoded an unexpected token plane: {array.shape} {array.dtype}")
    decoded_record = _atomic_raw_array(retained / "decoded_spatial_tokens.u8", array)
    decoded_sha = jo1.raw_array_sha256(array)
    if decoded_record["sha256"] != decoded_sha:
        raise CP5VError("persisted adapted-runtime token plane differs from its in-memory payload")
    shipped = primary_root / "receiver_state/decoded_spatial_tokens.shipped.bin"
    shipped_record = jo1.file_record(shipped)
    canonical_matches_shipped = decoded_record == {
        **shipped_record,
        "path": decoded_record["path"],
    }
    result = {
        "schema": "ddm_cp5v_adapted_runtime_decode.v1",
        "axis": AXIS,
        "archive": jo1.file_record(archive_path),
        "runtime_root": str(runtime_root.resolve()),
        "adapted_runtime_module": jo1.file_record(Path(f26.__file__).resolve()),
        "adapted_runtime_dependency": dependency,
        "rc64_source": jo1.file_record(runtime_root / "runtime/entropy/rc64_backend.c"),
        "rc64_library": jo1.file_record(rc64_library),
        "canonical_reader": "runtime.f26_inflate.decode_production_tokens",
        "cpr1_root": str((runtime_root / "cpr1").resolve()),
        "decoded_tokens": decoded_record,
        "decoded_token_raw_sha256": decoded_sha,
        "canonical_reader_report": token_report,
        "jo1_shipped_backend_tokens": shipped_record,
        "canonical_reader_match_to_jo1_shipped_backend": canonical_matches_shipped,
        "scorer_run": False,
        "score_claim": False,
        "all_payloads_retained": True,
    }
    jo1.atomic_json(result_path, result)
    if not canonical_matches_shipped:
        jo1.atomic_json(
            output / "FALSIFIER.json",
            {
                **result,
                "status": "FIRED",
                "reason": "adapted canonical reader differs from JO1 shipped receiver backend",
            },
        )
        raise CP5VError("receiver-path falsifier fired; retained FALSIFIER.json")
    return result


def _actual_token_diffs(
    base: np.ndarray,
    candidate: np.ndarray,
) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for frame in range(base.shape[0]):
        base_flat = np.asarray(base[frame]).reshape(-1)
        candidate_flat = np.asarray(candidate[frame]).reshape(-1)
        for index in np.flatnonzero(base_flat != candidate_flat).tolist():
            rows.append(
                {
                    "frame": frame,
                    "index": int(index),
                    "y": int(index) // jo1.WIDTH,
                    "x": int(index) % jo1.WIDTH,
                    "source_class": int(base_flat[index]),
                    "target_class": int(candidate_flat[index]),
                }
            )
    return rows


def _expected_token_diffs(applications: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for application in applications:
        for index in application["indices"]:
            rows.append(
                {
                    "proposal_id": application["proposal_id"],
                    "frame": int(application["frame"]),
                    "index": int(index),
                    "y": int(index) // jo1.WIDTH,
                    "x": int(index) % jo1.WIDTH,
                    "source_class": int(application["source_class"]),
                    "target_class": int(application["target_class"]),
                    "event_payload": application["payload"],
                }
            )
    return rows


def projected_score(delta_bytes: int) -> float:
    return jo1.BASE_SCORE - EXPECTED_SEG_SCORE_GAIN + 25.0 * delta_bytes / jo1.RATE_DENOMINATOR


def exact_eval_commands(archive: dict[str, Any], submission_dir: Path) -> dict[str, str]:
    lane_id = "lane_ddm_cp5v_validated_five_contest_cuda_20260812"
    job_id = "ddm_cp5v_validated_five_t4_20260812"
    output = "/Volumes/VertigoDataTier/pact/ddm_cp5v_20260812/main_t4"
    claim = (
        ".venv/bin/python tools/claim_lane_dispatch.py claim "
        f"--lane-id {lane_id} --platform modal --instance-job-id {job_id} --agent MAIN "
        "--status active_exact_eval_spawning --notes 'CP5V five n600-validated EC1 events; "
        f"sole contest-CUDA row; validate {archive['sha256']} at {archive['bytes']} B'"
    )
    dispatch = (
        ".venv/bin/modal run --detach experiments/modal_auth_eval.py::main "
        f"--archive {archive['path']} --output-dir {output} "
        f"--expected-archive-sha256 {archive['sha256']} --submission-dir {submission_dir.resolve()} "
        "--inflate-sh inflate.sh --gpu T4 --scorer-device cuda "
        f"--expected-runtime-tree-sha256 auto --lane-id {lane_id} --instance-job-id {job_id} "
        "--claim-agent MAIN --claim-policy require_active "
        "--single-axis-waiver-reason 'CP135 F26 family is CUDA-locked; contest-CPU refused by vehicle precedent' "
        "--detach --provider-detach-ack"
    )
    recover = f".venv/bin/python tools/recover_modal_auth_eval.py --output-dir {output}"
    return {"claim": claim, "dispatch": dispatch, "recover": recover}


def finalize(output: Path) -> dict[str, Any]:
    compose_result = json.loads((output / "20_COMPOSE_RESULT.json").read_text())
    decode_result = json.loads((output / "retained/adapted_runtime_decode/TOKEN_DECODE_RESULT.json").read_text())
    rows = load_census_rows()
    rows_by_id = {str(row["proposal_id"]): row for row in rows}
    primary_root = jo1.candidate_root(output, CANDIDATE_NAME)
    applications = json.loads((primary_root / "EVENT_APPLICATIONS.json").read_text())["rows"]
    expected = _expected_token_diffs(applications)
    base = np.memmap(
        jo1.BASE_SPATIAL,
        mode="r",
        dtype=np.uint8,
        shape=(jo1.FRAMES, jo1.HEIGHT, jo1.WIDTH),
    )
    candidate = np.memmap(
        Path(decode_result["decoded_tokens"]["path"]),
        mode="r",
        dtype=np.uint8,
        shape=(jo1.FRAMES, jo1.HEIGHT, jo1.WIDTH),
    )
    actual = _actual_token_diffs(base, candidate)
    expected_projection = sorted(
        ({key: row[key] for key in ("frame", "index", "y", "x", "source_class", "target_class")} for row in expected),
        key=lambda row: (row["frame"], row["index"]),
    )
    exact_diff = actual == expected_projection and len(actual) == len(EVENT_IDS)
    receipt_dir = output / "retained/per_event_diff_receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    per_event = []
    for row in expected:
        census = rows_by_id[str(row["proposal_id"])]
        receipt = {
            "schema": "ddm_cp5v_per_event_diff.v1",
            **row,
            "base_value_matches_source": int(base[row["frame"]].reshape(-1)[row["index"]]) == row["source_class"],
            "candidate_value_matches_target": int(candidate[row["frame"]].reshape(-1)[row["index"]])
            == row["target_class"],
            "n600_validation": {
                "pair": int(census["pair"]),
                "net_flip_gain_base_minus_candidate": int(census["net_flip_gain_base_minus_candidate"]),
                "delta_d_pose_global_n600": float(census["delta_d_pose_global_n600"]),
                "downstream_selection_eligible": bool(census["downstream_selection_eligible"]),
            },
        }
        path = receipt_dir / f"{row['proposal_id']}.json"
        jo1.atomic_json(path, receipt)
        per_event.append(jo1.file_record(path))
    diff_result = {
        "schema": "ddm_cp5v_token_diff_result.v1",
        "base_tokens": jo1.file_record(jo1.BASE_SPATIAL),
        "candidate_tokens": decode_result["decoded_tokens"],
        "expected_diff_count": len(expected),
        "actual_diff_count": len(actual),
        "expected": expected_projection,
        "actual": actual,
        "exactly_five_event_cells_and_nowhere_else": exact_diff,
        "per_event_receipts": per_event,
    }
    jo1.atomic_json(output / "30_TOKEN_DIFF_RESULT.json", diff_result)
    delta_bytes = int(compose_result["delta_bytes_vs_cp135"])
    if not exact_diff or delta_bytes > MAX_DELTA_BYTES:
        jo1.atomic_json(
            output / "FALSIFIER.json",
            {
                "schema": "ddm_cp5v_falsifier.v1",
                "status": "FIRED",
                "delta_bytes": delta_bytes,
                "max_delta_bytes": MAX_DELTA_BYTES,
                "token_diff": diff_result,
            },
        )
        raise CP5VError("terminal CP5V falsifier fired; retained FALSIFIER.json")

    archive = compose_result["primary"]
    runtime = primary_root / "adapted_runtime"
    commands = exact_eval_commands(archive, runtime)
    arithmetic = {
        "base_score": jo1.BASE_SCORE,
        "validated_seg_score_gain": EXPECTED_SEG_SCORE_GAIN,
        "validated_flip_gain": EXPECTED_FLIP_GAIN,
        "delta_bytes": delta_bytes,
        "delta_rate_score": 25.0 * delta_bytes / jo1.RATE_DENOMINATOR,
        "projected_score_if_singleton_seg_gain_is_additive": projected_score(delta_bytes),
        "projection_authority": "not an exact score; the composed n600 row is unmeasured",
        "additivity_calibration": {
            "readout": "realized composed delta versus sum of five n600 singleton deltas",
            "affected_pairs": list(AFFECTED_PAIRS),
        },
    }
    final = {
        "schema": "ddm_cp5v_final_result.v1",
        "status": "READY_TO_FIRE",
        "axis": AXIS,
        "archive": archive,
        "runtime": str(runtime.resolve()),
        "independent_repeat": compose_result["independent_repeat"],
        "archive_repeat_byte_identical": compose_result["independent_archive_byte_identical"],
        "rc64_repeat_byte_identical": compose_result["independent_rc64_byte_identical"],
        "receiver_closed": True,
        "adapted_runtime_canonical_reader_closed": decode_result["canonical_reader_match_to_jo1_shipped_backend"],
        "decoded_tokens": decode_result["decoded_tokens"],
        "token_diff": jo1.file_record(output / "30_TOKEN_DIFF_RESULT.json"),
        "event_ids": list(EVENT_IDS),
        "arithmetic": arithmetic,
        "exact_eval": {
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "MAIN exact contest-CUDA scorer owner",
            "consumer_store": str(output / "main_t4"),
            "fire_trigger": (
                "all live Modal/scorer claims are terminal, MAIN owns the sole scorer lane, "
                "and the archive/runtime pins still match"
            ),
            "commands": commands,
        },
        "scorer_run": False,
        "score_claim": False,
        "all_payloads_retained": True,
    }
    jo1.atomic_json(output / "FINAL_RESULT.json", final)
    save_state(output, "complete", ["preflight", "compose", "decode", "finalize"])
    return final


def all_stages(output: Path) -> dict[str, Any]:
    preflight(output)
    save_state(output, "compose", ["preflight"])
    compose(output)
    save_state(output, "decode", ["preflight", "compose"])
    decode_adapted_runtime(output)
    save_state(output, "finalize", ["preflight", "compose", "decode"])
    return finalize(output)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("stage", choices=("preflight", "compose", "decode", "finalize", "all"))
    value.add_argument("--output", type=Path, default=OUTPUT)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.stage == "preflight":
        result = preflight(args.output)
    elif args.stage == "compose":
        preflight(args.output)
        result = compose(args.output)
    elif args.stage == "decode":
        result = decode_adapted_runtime(args.output)
    elif args.stage == "finalize":
        result = finalize(args.output)
    else:
        result = all_stages(args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
