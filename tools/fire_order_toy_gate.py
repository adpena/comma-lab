#!/usr/bin/env python3
"""Fail-closed TOY gate at the FIRE site (operator 2026-08-13 "No naive or toy ever").

The pk3 incident: a clean charter did not stop a toy model being born at BUILD time,
and the fire path had no gate — the toy-derived sealed order was one command from a
paid dispatch. This gate is the last line. MAIN runs it against a candidate STORE
before executing any sealed `exact_command_argv`.

REFUSES (rc=1) when either holds:
  1. Any JSON in the store binds the candidate to a toy: a "verdict_scope" or
     "schema"/"status"/marker field containing TOY-BRACKET / toy_bracket, OR an
     explicit FIRE_WITHHELD_* receipt.
  2. The candidate is MODEL-DERIVED (any store JSON carries model_derived=true or a
     generalization_gate field) and no generalization receipt with passed=true exists.
PASSES (rc=0) otherwise — including solve-exact / mechanical candidates that carry no
model layer at all.

Usage: fire_order_toy_gate.py STORE_DIR [--json]
Positive control: --self-test runs both refuse branches on synthetic stores.
"""
import argparse
import json
import pathlib
import sys
import tempfile

TOY_TOKENS = ("TOY-BRACKET", "TOY_BRACKET", "toy_bracket", "toy-bracket")


def scan(store: pathlib.Path) -> dict:
    """Coarse fail-closed scan. REFUSED means "toy markers present — adjudicate",
    not "never fire": honest toy-LABELED projections beside a real mechanical
    candidate over-trigger the substring branch by design. The precision
    instrument is ADJUDICATION_FIRE_OK.json — a typed MAIN receipt with a real
    reason (placeholders rejected) stating why the CANDIDATE itself is not
    toy-derived. Forward stores should declare model_derived +
    generalization_gate explicitly for the precise branch."""
    adjudication = None
    adj_path = store / "ADJUDICATION_FIRE_OK.json"
    if adj_path.is_file():
        try:
            doc = json.loads(adj_path.read_text())
            reason = str(doc.get("reason", "")).strip()
            if len(reason) >= 20 and "<" not in reason:
                adjudication = reason
        except (json.JSONDecodeError, OSError):
            pass
    findings: list[str] = []
    model_derived = False
    gate_passed = False
    for p in sorted(store.rglob("*.json")):
        if p.name.startswith("FIRE_WITHHELD"):
            findings.append(f"withheld receipt present: {p.name}")
            continue
        try:
            doc = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        text = json.dumps(doc)
        if any(t in text for t in TOY_TOKENS):
            findings.append(f"toy-bracket marker in {p.relative_to(store)}")
        if isinstance(doc, dict):
            if doc.get("model_derived") is True or "generalization_gate" in doc:
                model_derived = True
            gate = doc.get("generalization_gate") or {}
            if isinstance(gate, dict) and gate.get("passed") is True:
                gate_passed = True
    if model_derived and not gate_passed:
        findings.append("model-derived candidate lacks a passed generalization receipt")
    if findings and adjudication:
        verdict = "PASS_WITH_ADJUDICATION"
    elif findings:
        verdict = "REFUSED"
    else:
        verdict = "PASS"
    return {
        "schema": "fire_order_toy_gate.v1",
        "store": str(store),
        "verdict": verdict,
        "adjudication": adjudication,
        "findings": findings,
    }


def self_test() -> int:
    with tempfile.TemporaryDirectory() as td:
        s = pathlib.Path(td)
        (s / "a.json").write_text(json.dumps({"verdict_scope": "TOY-BRACKET Q3C1"}))
        assert scan(s)["verdict"] == "REFUSED", "toy-marker refuse branch failed"
    with tempfile.TemporaryDirectory() as td:
        s = pathlib.Path(td)
        (s / "b.json").write_text(json.dumps({"model_derived": True}))
        assert scan(s)["verdict"] == "REFUSED", "missing-gate refuse branch failed"
    with tempfile.TemporaryDirectory() as td:
        s = pathlib.Path(td)
        (s / "c.json").write_text(json.dumps(
            {"model_derived": True, "generalization_gate": {"passed": True}}))
        assert scan(s)["verdict"] == "PASS", "pass branch failed"
    print("self-test: all 3 branches OK (2 refuse + 1 pass)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("store", nargs="?")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.store:
        ap.error("store directory required (or --self-test)")
    result = scan(pathlib.Path(args.store))
    print(json.dumps(result, indent=1) if args.json else
          f"{result['verdict']}: {'; '.join(result['findings']) or 'no toy bindings found'}")
    return 1 if result["verdict"] == "REFUSED" else 0


if __name__ == "__main__":
    sys.exit(main())
