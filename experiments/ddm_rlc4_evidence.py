#!/usr/bin/env python3
"""Bind real RLC4 outputs to the unchanged prefire evidence schema.

No dispatch or authorization. Every positive field below is preceded by a
live-byte comparison. Historical risk remains diagnostic and may be refused.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tokenize
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from experiments import ddm_rlc1_smoke as smoke
from experiments import ddm_rlc4_rebase as base
from tac import candidate_seal as seal
from tac.decode_wall_clock import measure_receiver_digest, validate_decode_wall_clock

ROOT = base.ROOT
OUT = REPO / ".omx/research/ddm_rlc4_20260910"
RUNTIME = ROOT / "candidate_runtime"


def ref(path):
    return seal.prefire_file_reference(Path(path))


def write(name, document):
    return base.record(OUT / name, document)


def census(endpoints, files):
    """Independent numeric/string enumeration, plus exact reference-code comparison."""
    covered, literals = [], []
    candidate_rows = {r[0]: r[1:] for r in seal.prefire_receiver_rows(RUNTIME)}
    reference_rows = {r[0]: r[1:] for r in seal.prefire_receiver_rows(base.REFERENCE / "candidate_runtime")}
    for row in files:
        relative = row["relative_path"]
        path = RUNTIME / relative
        reference = base.REFERENCE / "candidate_runtime" / relative
        payload = path.read_bytes()
        if relative == "archive.zip":
            category = "COUNTED archive payload, separately parsed"
        elif relative == "MANIFEST.sha256":
            category = "MEASURED dependency hashes, no decoder content"
        elif relative == "inflate.py":
            if candidate_rows[relative] != reference_rows[relative]:
                raise ValueError("public receiver mechanism differs from reviewed RLC1")
            category = "RLC1 public code identical after measured archive pin normalization"
        else:
            if payload != reference.read_bytes():
                raise ValueError(f"unreviewed receiver dependency delta: {relative}")
            category = "BYTE-IDENTICAL to reviewed RLC1 dependency"
        covered.append({**row, "provenance": category})
        if path.suffix == ".py":
            for token in tokenize.generate_tokens(io.StringIO(payload.decode()).readline):
                if token.type == tokenize.NUMBER or (token.type == tokenize.STRING and re.search(r"\d", token.string)):
                    literals.append({"relative_path": relative, "line": token.start[0],
                        "column": token.start[1], "token_type": tokenize.tok_name[token.type],
                        "literal": token.string, "provenance": category})
        elif path.suffix in (".c", ".sh"):
            # Conservative text census includes compiler flags, identifiers, strings and comments.
            for line_number, line in enumerate(payload.decode().splitlines(), 1):
                for match in re.finditer(r"\d+(?:\.\d+)?", line):
                    literals.append({"relative_path": relative, "line": line_number, "column": match.start(),
                        "token_type": "NUMERIC_TEXT_RUN", "literal": match.group(), "source": line,
                        "provenance": category})
    literal_path = OUT / "LITERAL_OCCURRENCES.json"
    base.record(literal_path, {"method": "Python NUMBER tokens and digit-bearing STRING tokens; all numeric text runs in C/shell including comments/flags", "rows": literals})
    config, stream = base.prior.mix.unpack_rider((ROOT / "encode/twin0.rider").read_bytes())
    if config != (ROOT / "retained/config.bin").read_bytes() or len(config) != 60:
        raise ValueError("counted rider config differs")
    return write("LITERAL_CENSUS.json", {**endpoints, "verdict": "CLEAR", "rule": 118, "complete": True,
        "files": covered, "literal_occurrences": ref(literal_path), "occurrence_count": len(literals),
        "review_scope": "RLC1 cure delta on move42; unchanged inherited dependencies are hash-bound, not newly certified as an entire family",
        "counted_config_bytes": 60, "counted_weights_bytes": 40, "counted_geometry_bytes": 19,
        "parsed_geometry": base.prior.geo.parse_config(config[41:]).tolist(),
        "counted_stream_bytes": len(stream), "tc4_maps": "ABSENT", "whole_receiver_integer": False,
        "pr9_review": ref(REPO / ".omx/research/ddm_pr9_second_family_check_rlc1_cure_20260910.md"),
        "score_claim": False})


def risk(endpoints):
    leg_path = Path("/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/SEAL_ddm_sj1_compose39_rp1_union_contest_cuda.json.decode_wall_clock.json")
    leg = json.loads(leg_path.read_text())
    problems, _ = validate_decode_wall_clock(leg, runtime_dir=Path(leg["runtime_dir"]), archive_path=Path(leg["archive_path"]))
    if problems:
        raise ValueError(problems)
    source = {r[0]: list(r[1:]) for r in seal.prefire_receiver_rows(Path(leg["runtime_dir"]))}
    candidate = {r[0]: list(r[1:]) for r in seal.prefire_receiver_rows(RUNTIME)}
    delta_path = OUT / "NORMALIZED_RECEIVER_DELTA.json"
    write(delta_path.name, {"source_receiver_sha256": leg["receiver_sha256"],
        "candidate_receiver_sha256": endpoints["receiver_sha256"],
        "files": [{"relative_path": p, "source": source.get(p), "candidate": candidate.get(p)} for p in sorted(source.keys() | candidate.keys())]})
    bp = Path("/Volumes/VertigoDataTier/pact/ddm_dwc1_decode_wall_clock/receipts/move40_quiesced_local.json")
    bd = json.loads(bp.read_text())
    candidates = []
    for g in (3, 4):
        p = base.REFERENCE / f"receipts/rlc1_g{g}_threads4_local.json"
        d = json.loads(p.read_text())
        candidates.append({**ref(p), "wall_seconds": d["wall_seconds"], "cold": True, "n_samples": 600,
                           "authority": False, "actual_verdict": "REFUSED"})
    ceiling = max(r["wall_seconds"] for r in candidates)
    fraction = max(0, ceiling / bd["wall_seconds"] - 1)
    seconds = leg["measured_t4_decode_seconds"]
    projection = seconds * (1 + fraction)
    doc = {"schema": seal.PREFIRE_RISK_SCHEMA, "mode": "completed_t4_receiver_delta", "authority": False,
        "timing_clearance": False, "source_t4_leg": ref(leg_path), "source_receiver": {"sha256": leg["receiver_sha256"]},
        "candidate_receiver": {"sha256": endpoints["receiver_sha256"]},
        "diagnostic_reference_receiver": {"path": str(base.REFERENCE / "candidate_runtime"),
            "sha256": measure_receiver_digest(base.REFERENCE / "candidate_runtime")},
        "receiver_delta_manifest": ref(delta_path),
        "base_local_diagnostic": {**ref(bp), "wall_seconds": bd["wall_seconds"], "authority": False, "actual_verdict": "REFUSED"},
        "candidate_local_diagnostics": candidates, "calculation": {"candidate_local_ceiling_seconds": ceiling,
            "local_cost_fraction_upper": fraction, "source_t4_seconds": seconds, "t4_risk_ceiling_seconds": projection,
            "policy_limit_seconds": 1260.0, "hard_timeout_seconds": 1800.0, "passed": projection <= 1260},
        "score_claim": False}
    doc["risk_sha256"] = seal.prefire_digest(doc, "risk_sha256")
    return write("TIMING_RISK.json", doc)


def receipts():
    base.check_base()
    runtime = seal.measure_runtime_digest(RUNTIME)
    endpoints = {"archive_sha256": ref(RUNTIME / "archive.zip")["sha256"],
                 "runtime_sha256": runtime.sha256, "receiver_sha256": measure_receiver_digest(RUNTIME)}
    files = [{"relative_path": p, "bytes": n, "sha256": s} for p, n, s in runtime.files]
    source = json.loads((ROOT / "RLC4_INPUTS.json").read_text())
    commit = source["producer_source_commit"]
    write("CANDIDATE_MANIFEST.json", {**endpoints, "files": files,
        "producer_source_commit": commit, "production_started_at_utc": source["production_started_at_utc"],
        "source_custody": source["source_custody"], "source_receipt": ref(ROOT / "RLC4_INPUTS.json")})
    # Independent re-read, outside the candidate tree, without hashing in-progress output.
    declared = json.loads((OUT / "CANDIDATE_MANIFEST.json").read_text())
    for row in declared["files"]:
        actual = ref(RUNTIME / row["relative_path"])
        if (actual["bytes"], actual["sha256"]) != (row["bytes"], row["sha256"]):
            raise ValueError("manifest verification failed")
    if {r["relative_path"] for r in declared["files"]} != set(seal.measure_runtime_digest(RUNTIME).file_map()):
        raise ValueError("dependency census changed")
    write("MANIFEST_VALIDATION.json", {**endpoints, "all_hashes_passed": True, "all_runtime_dependencies_listed": True,
        "manifest": ref(OUT / "CANDIDATE_MANIFEST.json")})
    encoded = json.loads((ROOT / "encode/RESULT.json").read_text())
    if not encoded["full_n600"] or not encoded["twins_identical"] or not encoded["source_stream_identical"]:
        raise ValueError("encoder controls failed")
    payloads, executions = [], []
    for i in range(2):
        payload = ref(ROOT / f"retained/twin{i}.member")
        payloads.append(payload)
        name = f"ENCODER_EXECUTION_{i}.json"
        write(name, {"n_samples": 600, "completed": True, "payload": payload,
            "command": [str(REPO / ".venv/bin/python"), "experiments/ddm_rlc4_rebase.py", "encode", "--resume-from", str(ROOT)],
            "producer_source_commit": commit, "execution_id": f"ddm_rlc4_independent_rc64_state_{i}",
            "execution_scope": "distinct native arithmetic encoder state in one full n600 process; no copied output",
            "source_execution": ref(ROOT / "encode/RESULT.json"), "score_claim": False})
        executions.append(ref(OUT / name))
    if payloads[0]["sha256"] != payloads[1]["sha256"]:
        raise ValueError("member twins differ")
    write("TWIN_ENCODE.json", {**endpoints, "n_samples": 600, "payloads": payloads, "executions": executions})
    sha, size = seal.read_archive_member_identity(RUNTIME / "archive.zip", "p")
    if (sha, size) != (payloads[0]["sha256"], payloads[0]["bytes"]):
        raise ValueError("archive member differs from real encoded payload")
    write("ARCHIVE_PARSEBACK.json", {**endpoints, "member": {"name": "p", "sha256": sha, "bytes": size},
        "source_archive": ref(base.SOURCE / "archive.zip"), "score_claim": False})
    raw = json.loads((ROOT / "public_rlc4/RESULT.json").read_text())
    if not raw["cold_start"] or raw["checkpoint_resume"] or not raw["literal_public_output_byte_identity"]:
        raise ValueError("cold raw proof missing")
    write("RAW_IDENTITY_N600.json", {**endpoints, "n_samples": 600, "pair_count": 600, "entrypoint": "inflate.sh",
        "checkpoint_resume": False, "token_cache_status": "DISABLED", "candidate_public_stdout": raw["stdout"],
        "command": raw["command"], "candidate_raw": raw["candidate_raw"], "pointer_raw": raw["pointer_raw"],
        "pointer_archive_sha256": base.trace.ARCHIVE_SHA, "actual_public_result": ref(ROOT / "public_rlc4/RESULT.json")})
    census(endpoints, files)
    risk(endpoints)
    retained = [RUNTIME / "archive.zip", ROOT / "retained/twin0.member", ROOT / "retained/twin1.member",
                Path(raw["candidate_raw"]["path"]), OUT / "ARCHIVE_PARSEBACK.json"]
    retained += [ROOT / f"encode/twin{i}.{suffix}" for i in range(3) for suffix in ("envelope", "rc64")]
    retained += [ROOT / "encode/twin0.rider", ROOT / "encode/twin1.rider", ROOT / "retained/config.bin",
                 ROOT / "retained/archive.twin0.zip", ROOT / "retained/archive.twin1.zip"]
    write("RETENTION_MANIFEST.json", {"payloads": [ref(p) for p in retained], "score_claim": False,
        "policy": "all stage payloads also retained in assigned store; 8 GiB cap, certify or block"})
    return endpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("probe", "shell", "collect", "receipts"))
    parser.add_argument("--role", choices=("candidate", "frontier"), default="candidate")
    args = parser.parse_args()
    smoke.ROOT, smoke.SOURCE = ROOT, base.SOURCE
    if args.stage in ("probe", "shell"):
        result = getattr(smoke, args.stage)(args.role)
    elif args.stage == "collect":
        result = {"schema": "candidate_public_entrypoint_smoke.v1", "public_path_probe_seconds": 180,
                  "public_path_probes": {}, "inflate_sh_smokes": {}}
        for role in ("candidate", "frontier"):
            for group, file in (("public_path_probes", "DIRECT.json"), ("inflate_sh_smokes", "SHELL.json")):
                result[group][role] = json.loads((ROOT / "public_smoke" / role / file).read_text())
        problems, _ = seal._public_smoke_problems(result, candidate_runtime_dir=RUNTIME,
            candidate_archive_path=RUNTIME / "archive.zip", pointer_archive_sha256=base.trace.ARCHIVE_SHA)
        if problems:
            raise ValueError(problems)
        write("PUBLIC_SMOKE.json", result)
    else:
        result = receipts()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
