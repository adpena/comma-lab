#!/usr/bin/env python3
"""Resume a terminal psa2 stage, preserving incomplete atomic writes first.

Refuses while ANY owned detached stage is active. Does not signal processes or
edit the source pins of running work. Each invocation resumes one failed stage;
the original receipt-driven continuation consumes that stage's new done receipt.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ddm_psa2_rgb_bias as p


def inventory() -> tuple[list[dict], list[dict]]:
    active, failed = [], []
    for path in p.STORE.rglob("launch_manifest.json"):
        if "recoveries" in path.relative_to(p.STORE).parts:
            continue
        manifest = p.load(path)
        receipt_path = Path(manifest["done_receipt_path"])
        if not receipt_path.exists():
            active.append({"manifest": str(path), "receipt": str(receipt_path)})
            continue
        receipt = p.load(receipt_path)
        if receipt.get("schema") != "detached_local_process_done.v2":
            raise ValueError("unrecognized done receipt")
        if receipt["rc"] != 0:
            failed.append({"manifest": manifest, "path": str(path), "receipt": p.fact(receipt_path)})
    return active, failed


def preserve_pending(root: Path, manifests: list[dict], source_pins: list[dict]) -> list[dict]:
    paths = [x for x in p.STORE.rglob("*.pending")
             if x.is_file() and "recoveries" not in x.parts
             and not any(part.startswith("._") for part in x.parts)]
    p.capacity(sum(x.stat().st_size for x in paths))
    records = []
    for index, path in enumerate(paths):
        before = p.fact(path)
        dest = root / "interrupted" / f"{index:04d}_{before['sha256']}"
        record = {"original": before, "destination": str(dest), "reason": "interrupted atomic stage output; complete checkpoints retained",
                  "reproducibility": {"seed": p.SEED, "source_pins": source_pins,
                                      "failed_stage_manifests": manifests, "base_binding": p.fact(p.STORE / "BINDING.json")},
                  "score_claim": False, "rebuildable_by": "same recorded argv using --resume-from and retained complete blocks"}
        p.save(root / "certificates" / f"{index:04d}.json", record)
        dest.parent.mkdir(parents=True, exist_ok=True)
        path.rename(dest)
        after = p.fact(dest)
        if (before["bytes"], before["sha256"]) != (after["bytes"], after["sha256"]):
            raise ValueError("interrupted payload custody changed")
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    args = parser.parse_args()
    if args.resume_from.resolve() != p.STORE.resolve():
        raise ValueError("wrong resume store")
    active, failed = inventory()
    if active:
        print(json.dumps({"status": "ACTIVE_JOBS_REFUSED", "active": active, "writes": 0, "signals": 0}))
        return 2
    if not failed:
        print(json.dumps({"status": "NO_FAILED_STAGE", "writes": 0}))
        return 0
    v2 = p.STORE / "FOLLOWTHROUGH_V2_BINDING.json"
    sources = p.load(v2 if v2.exists() else p.STORE / "FOLLOWTHROUGH_BINDING.json")["sources"]
    for source in sources:
        p.checked(source)
    if v2.exists():
        # V2 waits for this deliberately stale waiter to refuse before it can
        # launch a child. Restarting V1 could only fail the same old source pin.
        old = [x for x in failed if Path(x["manifest"]["done_receipt_path"]).name == "ddm_psa2_followthrough.done"]
        expected = "source or checkpoint drift: " + str(p.REPO / "experiments/ddm_psa2_resolve_price.py")
        if old and expected not in (p.STORE / "launch_followthrough/run.log").read_text():
            raise ValueError("unexpected original waiter failure; inspect before resume")
        failed = [x for x in failed if x not in old]
        if not failed:
            print(json.dumps({"status": "ONLY_SUPERSEDED_WAITER_FAILED", "writes": 0}))
            return 0

    def priority(row):
        argv = row["manifest"]["argv"]
        name = Path(row["manifest"]["done_receipt_path"]).name
        rank = 0 if "ddm_psa2_search_full" in name else 3 if "ddm_psa2_followthrough" in name else 1
        return rank, len(argv), name

    chosen = min(failed, key=priority)
    manifest = chosen["manifest"]
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    recovery = p.STORE / "recoveries" / stamp
    # The receipt proves terminal state; no live file or running source is touched.
    records = preserve_pending(recovery, [x["manifest"] for x in failed], sources)
    name = Path(manifest["done_receipt_path"]).name.removesuffix(".done")
    output = p.STORE / ("resume_" + name + "_" + stamp)
    argv = [sys.executable, "tools/launch_detached_process.py", "--output-dir", str(output),
            "--purpose", manifest["purpose"], "--authority", "ddm_psa2 crash resume after terminal receipts and payload custody",
            "--artifact-budget-gib", "0.5", "--done-receipt", name, "--receipt-supersede",
            "--env", "PYTHONDONTWRITEBYTECODE=1", "--", *manifest["argv"]]
    p.save(recovery / "RESUME_INTENT.json", {"argv": argv, "prior": chosen, "preserved": records, "sources": sources})
    result = subprocess.run(argv, cwd=p.REPO, text=True, capture_output=True, check=False)
    p.save(recovery / "RESUME_RESULT.json", {"rc": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    print(json.dumps({"status": "RESUME_LAUNCHED" if result.returncode == 0 else "RESUME_REFUSED",
                      "rc": result.returncode, "recovery": str(recovery), "receipt_name": name,
                      "next": "after this stage is terminal, run the same resume command to restart any failed continuation"}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
