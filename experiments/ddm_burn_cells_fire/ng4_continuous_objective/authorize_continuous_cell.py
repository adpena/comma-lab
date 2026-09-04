#!/usr/bin/env python3
"""MAIN authorize step for ng4's continuous-objective cell: bind fresh claim ids through the chain driver's OWN
functions (authorized_config / write_or_verify_authorized), never a hand-edited JSON.
Usage: authorize_continuous_cell.py --scorer-claim-id ID --metal-claim-id ID [--write]"""
import argparse, json, sys
from pathlib import Path
REPO = Path("/Users/adpena/Projects/pact")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments"))
import ddm_qbr1_cell_chain as chain  # noqa: E402

STORE = Path("/Volumes/APDataStore/pact/ddm_ng4_continuous_objective")
SEALED = STORE / "sealed_configs/seed_20260902_continuous_objective_control_native100.rerooted.json"
AUTH = STORE / "authorized_configs/seed_20260902_continuous_objective_control_native100.json"

ap = argparse.ArgumentParser(); ap.add_argument("--scorer-claim-id", required=True); ap.add_argument("--metal-claim-id", required=True); ap.add_argument("--write", action="store_true")
a = ap.parse_args()
sealed = json.loads(SEALED.read_text())
expected = chain.authorized_config(sealed, a.scorer_claim_id, a.metal_claim_id)
changed = sorted(k for k in set(sealed) | set(expected) if sealed.get(k) != expected.get(k))
print(json.dumps({"sealed": str(SEALED), "authorized": str(AUTH), "fields_changed": changed,
                  "scorer_lane": expected["scorer_lane"], "metal_lane": expected["metal_lane"], "launch_authorized": expected["launch_authorized"]}, indent=2))
if a.write:
    AUTH.parent.mkdir(parents=True, exist_ok=True)
    fact = chain.write_or_verify_authorized(AUTH, expected)
    print(json.dumps({"written": fact}, indent=2))
