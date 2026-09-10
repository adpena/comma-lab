#!/usr/bin/env python3
"""Bind the landed RLC3/RLC4 chain to move43; all payloads retained on SSD.

The inherited internal directory 'move42', receipt names RLC4_*, and public_rlc4
are compatibility names in a fresh move43 store, never reused move42 evidence.
No scorer, timing window, authorization, or dispatch is provided by this runner.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from experiments import ddm_rlc3_move42_trace as trace
from experiments import ddm_rlc4_rebase as base
from experiments import ddm_rlc4_public as public
from experiments import ddm_rlc4_evidence as evidence
from tac import candidate_seal as seal

ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43")
SOURCE = Path("/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/candidate/candidate_runtime")
OUT = REPO / ".omx/research/ddm_rlc5_20260910"
ARCHIVE_SHA = "7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e"
FIELD_SHA = "a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8"
RAW_SHA = "2b762eba4a20a315c104f8447d6ea0e604f73c3d8b8b69b3fc63b0fc792d59fc"
RISK_SHA = "9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890"


def configure():
    """Explicitly bind inherited producers before any preparation or execution."""
    trace.ROOT, trace.LIVE = ROOT, SOURCE
    trace.FIELD = SOURCE.parents[1] / "parseback/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
    trace.ARCHIVE_SHA, trace.FIELD_SHA = ARCHIVE_SHA, FIELD_SHA
    base.ROOT, base.SOURCE = ROOT, SOURCE
    base.BASE_BYTES, base.MAX_ARCHIVE_BYTES = 180466, 180435
    base.prior.ROOT = ROOT
    public.ROOT = ROOT
    public.SOURCE_RAW = SOURCE.parents[1] / "parseback/0.raw"
    public.SOURCE_RAW_SHA = RAW_SHA
    evidence.ROOT, evidence.OUT, evidence.RUNTIME = ROOT, OUT, ROOT / "candidate_runtime"
    evidence.ARM_ID, evidence.BASE_LABEL = "ddm_rlc5", "move43"
    evidence.ENCODER_COMMAND = [str(REPO / ".venv/bin/python"), str(Path(__file__)),
                                "encode", "--resume-from", str(ROOT)]
    evidence.smoke.ROOT, evidence.smoke.SOURCE = ROOT, SOURCE
    base.check_base()
    if seal.measure_prefire_risk_receiver_digest(base.REFERENCE / "candidate_runtime") != RISK_SHA:
        raise ValueError("AMENDED_REFERENCE_DRIFT: stop, do not patch contract")
    trace.storage(ROOT / "retained")
    binding = {"archive": base.fact(SOURCE / "archive.zip"), "field": base.fact(trace.FIELD),
               "base_bytes": 180466, "byte_gate": 180435, "source_raw_sha256": RAW_SHA,
               "producer_sources": [base.fact(Path(m.__file__)) for m in (trace, base, public, evidence)]
                   + [base.fact(Path(__file__))],
               "amended_freeze": base.fact(REPO / seal.PREFIRE_FREEZE),
               "score_claim": False}
    if binding["field"]["sha256"] != FIELD_SHA:
        raise ValueError("MOVE43_FIELD_DRIFT")
    trace.blob(ROOT / "retained/RLC5_BINDINGS.json", (json.dumps(binding, sort_keys=True, indent=2) + "\n").encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "trace", "encode", "stage", "public", "probe", "shell", "collect", "receipts"))
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=600)
    parser.add_argument("--role", choices=("candidate", "frontier"), default="candidate")
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT.resolve() or not 0 < args.stop_after <= 600:
        raise ValueError("wrong root or frame boundary")
    configure()
    if args.stage == "prepare":
        result = trace.prepare()
    elif args.stage == "trace":
        result = trace.run("trace", args.stop_after)
    elif args.stage == "encode":
        result = base.encode(args.stop_after)
    elif args.stage == "stage":
        result = base.stage()
        if result.get("verdict") == "STOP":
            raise ValueError("BYTE_GATE_STOP: retain both real encodes")
        rows = seal.prefire_risk_receiver_rows(ROOT / "candidate_runtime")
        if len(rows) != 49 or seal.measure_prefire_risk_receiver_digest(ROOT / "candidate_runtime") != RISK_SHA:
            raise ValueError("AMENDED_CANDIDATE_DRIFT: stop, do not patch contract")
        evidence.write("RISK_RECEIVER_ROWS.json", rows)
    elif args.stage == "public":
        result = public.run("rlc4")
    elif args.stage in ("probe", "shell"):
        result = getattr(evidence.smoke, args.stage)(args.role)
    elif args.stage == "receipts":
        result = evidence.receipts()
    else:
        result = {"schema": "candidate_public_entrypoint_smoke.v1", "public_path_probe_seconds": 180,
                  "public_path_probes": {}, "inflate_sh_smokes": {}}
        for role in ("candidate", "frontier"):
            for group, name in (("public_path_probes", "DIRECT.json"), ("inflate_sh_smokes", "SHELL.json")):
                result[group][role] = json.loads((ROOT / "public_smoke" / role / name).read_text())
        problems, _ = seal._public_smoke_problems(result, candidate_runtime_dir=ROOT / "candidate_runtime",
            candidate_archive_path=ROOT / "candidate_runtime/archive.zip", pointer_archive_sha256=ARCHIVE_SHA)
        if problems:
            raise ValueError(problems)
        evidence.write("PUBLIC_SMOKE.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
