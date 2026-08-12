#!/usr/bin/env python
"""ddm_t0 terminal intake binder — execute runbook T0 on the PASS-4 terminal state.

Provenance chain:
  - Runbook: .omx/research/ddm_t0r1_terminal_day_runbook_20260811.md (+ its
    TRANCHE-STOP ADDENDUM, applied here with pass_04 substituted for pass_06:
    the solve ended at pass 4 via the launcher's 24h timeout, adjudicated
    terminal in the js1 skeleton commit beb39b41a9).
  - Rehearsal template: /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/
    t0_rehearsal_pass03/ (6 bound / 1 typed-unresolved / m37 3-of-3, ddm_t0r1).
  - Binder callables: tac.witness_dsl.hr1_prestage (bind_existing_file,
    unresolved_terminal_binding, assert_same_parent_freshness) — the same
    calls the rehearsal exercised.
  - Extraction adapter: experiments/ddm_ps135_pose_resolve.py
    parse_candidate_archive (lossless CX2 decode + TM1 split + section
    decomposition, REFUSES on any section drift from the pinned LC2 source).

Scorer-free. Never sets execution_allowed=true. Never launches anything.
The §5 activation join is recorded PENDING_T2_COMPILE: its input
(compiled_terminal_config.json) is produced by the js1 reseal (T2); the
rehearsal's 9/9 clean join stands as the positive control until then.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.witness_dsl.hr1_prestage import (  # noqa: E402
    FileBindingRequest,
    Hr1PrestageError,
    assert_same_parent_freshness,
    bind_existing_file,
    stream_sha256,
    unresolved_terminal_binding,
)

T0_SOURCE = Path("/Volumes/VertigoDataTier/pact/ddm_ps135_20260810")
T0_STORE = Path("/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hr1_preflight")
PASS_DIR = T0_SOURCE / "leg_a" / "passes" / "pass_04"
SELECTED = PASS_DIR / "selected"
POLICY_RECEIPT = T0_SOURCE / "leg_a" / "TERMINAL_BY_POLICY.json"
STATE_PATH = T0_SOURCE / "leg_a" / "state.json"
SAFE_RUN = T0_SOURCE / "gen3_resume" / "leg_a_resume.safe_run.json"
MAP_STORE = Path("/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/mixed_precision")
MAP_NPZ = MAP_STORE / "lc2_int12_pose_sensitivity_map.npz"
MAP_DISPOSITION = MAP_STORE / "STAGE_C_DISPOSITION.json"
RUNNER = REPO / "experiments" / "ddm_ps135_pose_resolve.py"

TERMINAL_PASS = 4
# Custody pins from the committed t1r1 memo (ddm_t1r1_container_build_rehearsal_20260812.md)
# cross-checked against pass_04/selected/receipt.json this session.
PIN_ARCHIVE_SHA = "e269d1ffbe0bf56ec8471a6869b7ec081f3de07e852b193aa251a963c543becb"
PIN_CARRIER_SHA = "4c1a65c7f3a9bfa1b0f7677494ddbfdad87881fe0f4b78613893bd555f725ef2"
PIN_COEFFS_SHA = "da9bba74fdaadc8110b9eb0614decb6d3a5caa076a03b01eee5647d32c37590e"
# Section pins: invariant across passes (only the carrier changes); values from
# the pass-03 rehearsal extraction AND pass_04/selected/receipt.json semantic sha.
PIN_RENDERER_SHA = "9b98360bd56918b5a414ace375c29790b7fe9f7f55cf423c0564ef4e62a39b99"
PIN_HPAC_SHA = "b07fff73fac41c5fec2d8acbfd7c43c518852696f18d95cf7465fc6ed7510b58"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest, _ = stream_sha256(path)
    return digest


def file_record(path: Path) -> dict:
    digest, size = stream_sha256(path)
    return {"path": str(path), "bytes": size, "sha256": digest}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def refuse(message: str) -> None:
    raise SystemExit(f"T0 REFUSE: {message}")


def section_1_pins() -> dict:
    """Runbook §1 with tranche substitutions (RESULT.json -> TERMINAL_BY_POLICY.json)."""
    for path in (POLICY_RECEIPT, STATE_PATH, SAFE_RUN, SELECTED / "archive.zip"):
        if not path.is_file():
            refuse(f"§1 missing terminal pin: {path}")
    policy = json.loads(POLICY_RECEIPT.read_text(encoding="utf-8"))
    if policy.get("resumable") is not True:
        refuse("§1 policy receipt does not attest resumability")
    if int(policy.get("passes_completed") or 0) != TERMINAL_PASS:
        refuse(
            f"§1 policy receipt passes_completed={policy.get('passes_completed')} "
            f"!= terminal pass {TERMINAL_PASS}"
        )
    safe_run = json.loads(SAFE_RUN.read_text(encoding="utf-8"))
    if safe_run.get("status") == "running":
        refuse("§1 safe-run receipt still running")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if int(state.get("passes_completed") or 0) != TERMINAL_PASS:
        refuse("§1 state.json passes_completed drifted from the terminal pass")
    return {
        "policy_receipt": file_record(POLICY_RECEIPT),
        "state": file_record(STATE_PATH),
        "safe_run": file_record(SAFE_RUN),
        "safe_run_status": safe_run.get("status"),
        "passes_completed": TERMINAL_PASS,
    }


def section_2_producers() -> dict:
    required = {
        "archive": SELECTED / "archive.zip",
        "carrier": SELECTED / "carrier.cpr1",
        "coefficients": SELECTED / "coefficients.int16.npy",
        "pass_receipt": SELECTED / "receipt.json",
        "global_exact_refresh": SELECTED / "global_exact_refresh.json",
        "policy_receipt": POLICY_RECEIPT,
        "sensitivity_map": MAP_NPZ,
        "stage_c_disposition": MAP_DISPOSITION,
    }
    for label, path in required.items():
        if not path.is_file():
            refuse(f"§2 missing terminal producer {label}: {path}")
    return {label: file_record(path) for label, path in required.items()}


def load_runner():
    spec = importlib.util.spec_from_file_location("ps135_resolve", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ps135_resolve"] = module
    spec.loader.exec_module(module)
    return module


def extract_and_materialize(objects_dir: Path) -> dict:
    """Adapter rows: lossless CX2/TM1 extraction, verified against the LC2 source."""
    ps135 = load_runner()
    source = ps135.load_lc2_source()
    carrier = (SELECTED / "carrier.cpr1").read_bytes()
    archive = (SELECTED / "archive.zip").read_bytes()
    parseback = ps135.parse_candidate_archive(archive, carrier, source)
    if parseback["archive_sha256"] != PIN_ARCHIVE_SHA:
        refuse("adapter: parsed archive sha differs from the terminal pin")
    state = ps135.decode_carrier(carrier)
    coefficients = np.load(SELECTED / "coefficients.int16.npy", allow_pickle=False)
    if not np.array_equal(coefficients, state.codes):
        refuse("adapter: direct coefficients differ from CX2/TM1 carrier parse-back")

    objects_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SELECTED / "archive.zip", objects_dir / "terminal_archive.zip")
    shutil.copy2(SELECTED / "carrier.cpr1", objects_dir / "terminal_carrier.cpr1")
    shutil.copy2(
        SELECTED / "coefficients.int16.npy",
        objects_dir / "terminal_coefficients.int16.npy",
    )
    # parse_candidate_archive REFUSES unless the archive's semantic and HPAC
    # sections byte-equal the pinned LC2 source, so source bytes ARE the
    # archive-embedded sections.
    (objects_dir / "terminal_renderer.semantic.bin").write_bytes(source.semantic)
    (objects_dir / "terminal_probability_object.hpac.bin").write_bytes(source.hpac_base)
    return {
        "parseback": {
            key: parseback[key]
            for key in (
                "archive_bytes",
                "archive_sha256",
                "semantic_bytes",
                "semantic_sha256",
                "carrier_bytes",
                "carrier_sha256",
                "tokens_sha256",
                "bound_semantic_surface",
            )
        },
        "coefficients_equal_parsed_codes": True,
    }


def build_convergence_receipt(objects_dir: Path) -> dict:
    """Ordered projection of complete pass receipts, ending at the terminal archive."""
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    history = state.get("history", [])
    rows = []
    for index in range(1, TERMINAL_PASS + 1):
        receipt_path = (
            T0_SOURCE / "leg_a" / "passes" / f"pass_{index:02d}" / "selected" / "receipt.json"
        )
        if not receipt_path.is_file():
            refuse(f"convergence: missing pass receipt {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "pass": index,
                "receipt": file_record(receipt_path),
                "selected_archive_sha256": receipt["parseback"]["archive_sha256"],
            }
        )
    if rows[-1]["selected_archive_sha256"] != PIN_ARCHIVE_SHA:
        refuse("convergence: final pass receipt does not end at the terminal archive")
    payload = {
        "schema": "ddm_t0_terminal_convergence_receipt.v1",
        "written_at_utc": utc_now(),
        "terminal_pass": TERMINAL_PASS,
        "terminal_archive_sha256": PIN_ARCHIVE_SHA,
        "passes": rows,
        "state_history": history,
        "policy_receipt_sha256": sha256_file(POLICY_RECEIPT),
    }
    atomic_json(objects_dir / "terminal_convergence_receipt.json", payload)
    return payload


def bind_roles(objects_dir: Path) -> tuple[list, dict]:
    bindings = [
        bind_existing_file(
            FileBindingRequest(
                "terminal_archive",
                objects_dir / "terminal_archive.zip",
                expected_sha256=PIN_ARCHIVE_SHA,
            )
        ),
        bind_existing_file(
            FileBindingRequest(
                "terminal_renderer",
                objects_dir / "terminal_renderer.semantic.bin",
                expected_sha256=PIN_RENDERER_SHA,
            )
        ),
        bind_existing_file(
            FileBindingRequest(
                "terminal_carrier",
                objects_dir / "terminal_carrier.cpr1",
                expected_sha256=PIN_CARRIER_SHA,
            )
        ),
        bind_existing_file(
            FileBindingRequest(
                "terminal_coefficients",
                objects_dir / "terminal_coefficients.int16.npy",
                expected_sha256=PIN_COEFFS_SHA,
            )
        ),
        bind_existing_file(
            FileBindingRequest(
                "terminal_probability_object",
                objects_dir / "terminal_probability_object.hpac.bin",
                expected_sha256=PIN_HPAC_SHA,
            )
        ),
        bind_existing_file(
            FileBindingRequest(
                "terminal_convergence_receipt",
                objects_dir / "terminal_convergence_receipt.json",
            )
        ),
    ]
    # Tranche addendum: the map role is unresolved BY POLICY, planning-band,
    # retired consumer — never silently bound, never silently dropped.
    map_binding = unresolved_terminal_binding(
        "terminal_sensitivity_map",
        resolution_trigger=(
            "unresolved_by_policy: emitted map parent is the pass-4 INPUT archive "
            "(pass-3 selected), stale by one generation at the tranche stop; binds "
            "PLANNING_BAND with consumer_status=RETIRED_PER_PZ4A_20260811; a future "
            "resumed solve reaching a natural exit may bind it terminal"
        ),
    )
    return bindings, map_binding


def m37_receipts(objects_dir: Path) -> dict:
    """Three same-parent freshness receipts; the map gets an explicit refusal record."""
    pass_receipt = json.loads((SELECTED / "receipt.json").read_text(encoding="utf-8"))
    refresh = json.loads(
        (SELECTED / "global_exact_refresh.json").read_text(encoding="utf-8")
    )
    archive_sha = sha256_file(objects_dir / "terminal_archive.zip")
    coeff_sha = sha256_file(objects_dir / "terminal_coefficients.int16.npy")

    receipts = [
        {
            "name": "selected_archive_receipt",
            **assert_same_parent_freshness(
                object_kind="selector",
                producer_parent_sha=pass_receipt["parseback"]["archive_sha256"],
                consumer_parent_sha=archive_sha,
            ).to_dict(),
        },
        {
            "name": "global_exact_refresh_selector",
            **assert_same_parent_freshness(
                object_kind="selector",
                producer_parent_sha=refresh["selected_archive_sha256"],
                consumer_parent_sha=archive_sha,
            ).to_dict(),
        },
        {
            "name": "coefficient_fit",
            **assert_same_parent_freshness(
                object_kind="fit",
                producer_parent_sha=pass_receipt["records"]["coefficients"]["sha256"],
                consumer_parent_sha=coeff_sha,
            ).to_dict(),
        },
    ]

    map_parent = sha256_file(PASS_DIR / "input_archive.zip")
    try:
        assert_same_parent_freshness(
            object_kind="map",
            producer_parent_sha=map_parent,
            consumer_parent_sha=archive_sha,
        )
        refuse("map freshness unexpectedly PASSED — custody model is wrong, stop")
    except Hr1PrestageError as exc:
        map_refusal = {
            "name": "terminal_sensitivity_map",
            "object_kind": "map",
            "producer_parent_sha": map_parent,
            "consumer_parent_sha": archive_sha,
            "freshness_ok": False,
            "refusal": str(exc),
            "policy_binding": {
                "band": "PLANNING_BAND",
                "consumer_status": "RETIRED_PER_PZ4A_20260811",
                "map": file_record(MAP_NPZ),
                "disposition": file_record(MAP_DISPOSITION),
            },
        }
    return {
        "complete": True,
        "denominator": 3,
        "passes": 3,
        "receipts": receipts,
        "map_refusal_receipt": map_refusal,
    }


def main() -> int:
    pins = section_1_pins()
    producers = section_2_producers()
    objects_dir = T0_STORE / "content_bindings" / "objects"
    adapter = extract_and_materialize(objects_dir)
    convergence = build_convergence_receipt(objects_dir)
    bindings, map_binding = bind_roles(objects_dir)
    receipts = m37_receipts(objects_dir)

    manifest = {
        "schema": "ddm_t0_terminal_content_bindings.v1",
        "written_at_utc": utc_now(),
        "terminal_label": "TERMINAL_AT_PASS_04_BY_LAUNCHER_TIMEOUT_ADJUDICATED",
        "score_claim": False,
        "scorer_run": False,
        "execution_allowed": False,
        "bound_role_count": len(bindings),
        "unresolved_role_count": 1,
        "unresolved_by_policy": ["terminal_sensitivity_map"],
        "bindings": [binding.to_dict() for binding in bindings],
        "map_binding": map_binding.to_dict(),
        "section_1_pins": pins,
        "section_2_producers": producers,
        "adapter": adapter,
        "convergence_passes": convergence["terminal_pass"],
        "activation_join": {
            "status": "PENDING_T2_COMPILE",
            "reason": (
                "compiled_terminal_config.json is produced by the js1 reseal (T2); "
                "the rehearsal join (9/9 levers, rc=0) is the standing positive "
                "control; the join fires as the FIRST act after the reseal compiles, "
                "before any launch"
            ),
        },
    }
    atomic_json(T0_STORE / "content_bindings" / "terminal_content_bindings.json", manifest)
    atomic_json(T0_STORE / "content_bindings" / "terminal_m37_receipts.json", receipts)

    # §6 amended admit checks (tranche addendum: SIX bound + ONE policy-unresolved).
    admit = (
        manifest["bound_role_count"] == 6
        and manifest["unresolved_role_count"] == 1
        and manifest["execution_allowed"] is False
        and receipts["passes"] == 3
        and receipts["map_refusal_receipt"]["freshness_ok"] is False
    )
    print(json.dumps({
        "t0_admit": admit,
        "bound_roles": manifest["bound_role_count"],
        "unresolved_by_policy": manifest["unresolved_by_policy"],
        "m37_passes": f"{receipts['passes']}/{receipts['denominator']}",
        "activation_join": manifest["activation_join"]["status"],
        "store": str(T0_STORE / "content_bindings"),
    }, indent=2))
    return 0 if admit else 5


if __name__ == "__main__":
    raise SystemExit(main())
