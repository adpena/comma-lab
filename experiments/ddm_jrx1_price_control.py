#!/usr/bin/env python3
"""Actual twin RLC1 prices for the fixed jrx1 field-control sample.

Reuse pd4's six-proposal sheet ranking, then charge each chosen pair with its
own complete real encode. Sheets are ranking observations, never byte charges.
The priced archives hold the shipped carrier; resolved pose remains a diagnostic
leg, as in pd4. No candidate, renderer change or joint measurement is made here.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import io
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "experiments"), str(REPO), str(REPO / "src")]

import ddm_jrx1_field_control as control
import ddm_pd4_pose_directed_pass4 as pd4
import numpy as np

STORE = control.STORE
ROOT = STORE / "prices"
REFERENCE = Path("/Volumes/APDataStore/pact/ddm_jrd1/rlc1_price/INPUTS.json")
TAIL_CONTROL = REPO / ".omx/research/ddm_jrd1_20260916/TAIL_CONTROL.json"
TAIL_CONTROL_SHA = "7666c1059373b493cfc05c2dc9c943f4d4e93cf383b6efad045b96a898480b6d"
ENCODE_BUDGET = 256 << 20


def load(path: Path):
    return json.loads(path.read_text())


def checked_fact(fact: dict) -> None:
    if control.custody.fact(Path(fact["path"])) != fact:
        raise ValueError(f"custody changed: {fact['path']}")


def save(path: Path, value) -> dict:
    return control.save_json(path, value)


def verify_live(inputs: dict) -> None:
    pointer = load(REPO / ".omx/state/canonical_frontier_pointer.json")
    if pointer["our_local_frontier_contest_cuda"]["archive_sha256"] != inputs["pointer_archive"]["sha256"]:
        raise ValueError("POINTER_MOVED: stop and re-pin")
    for fact in inputs["runtime_sources"].values():
        checked_fact(fact)
    checked_fact(inputs["fields"]["control"]["u8"])


def job_bytes(root: Path) -> int:
    while True:
        try:
            return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        except FileNotFoundError:
            continue


def reserve_encode(root: Path) -> None:
    # Searches use the same store and must be terminal before these reservations
    # are meaningful. Preparation itself never runs concurrently with encoders.
    for pair in load(control.SAMPLE)["pairs"]:
        if not (control.SEARCH / f"pair_{pair:03d}/DONE.json").exists():
            raise ValueError("finish all fixed-sample search stages before pricing")
    with (STORE / ".custody.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        pending = 0
        for path in ROOT.glob("*/BUDGET.json"):
            if not (path.parent / "DONE.json").exists():
                pending += max(0, load(path)["maximum_job_bytes"] - job_bytes(path.parent))
        own = root / "BUDGET.json"
        extra = 0 if own.exists() else max(0, ENCODE_BUDGET - job_bytes(root))
        control.custody.check_capacity(pending + extra + (4 << 20))
        # We already hold the custody lock, so call its unwrapped writer.
        control._serial_retain(own, (json.dumps({"maximum_job_bytes": ENCODE_BUDGET,
            "basis": "jrd1 complete encoder controls used less than 256MiB including 118MB raw field",
            "root": str(root)}, sort_keys=True) + "\n").encode())


def field_from_rows(rows: list[dict]) -> np.ndarray:
    entry = load(REFERENCE)["fields"]["control"]["u8"]
    checked_fact(entry)
    field = np.fromfile(entry["path"], dtype=np.uint8).reshape(600, 384, 512)
    for row in rows:
        plane = field[int(row["pair"])]
        for r, c, old, new in pd4.row_edits(row):
            if int(plane[r, c]) != old:
                raise ValueError("proposal source symbol differs from move52")
            plane[r, c] = new
    return field


def prepare_job(name: str, rows: list[dict]) -> None:
    if not name.isidentifier():
        raise ValueError("job name must be an identifier")
    root = ROOT / name
    if (root / "JOB.json").exists():
        old = load(root / "JOB.json")
        if old["rows"] != rows:
            raise ValueError("attempt to replace an immutable pricing job")
        checked_fact(old["field_npz"])
        return
    control.custody.check_capacity(8 << 20)
    field = field_from_rows(rows)
    buf = io.BytesIO()
    np.savez_compressed(buf, **{str(p): field[p] for p in range(600)})
    npz = control.locked_retain(root / "field.npz", buf.getvalue())
    raw = {"path": str(root / "bulk/field.u8"), "bytes": int(field.nbytes),
           "sha256": control.digest(memoryview(field))}
    with np.load(npz["path"], allow_pickle=False) as data:
        if any(not np.array_equal(data[str(p)], field[p]) for p in range(600)):
            raise ValueError("retained npz does not reproduce the real field")
    original = load(REFERENCE)
    for fact in original["runtime_sources"].values():
        checked_fact(fact)
    runtime = STORE / "price_runtime"
    owned_runtime = []
    for relative, fact in original["runtime_sources"].items():
        owned_runtime.append(control.locked_retain(runtime / relative, Path(fact["path"]).read_bytes()))
    inputs = copy.deepcopy(original)
    inputs["runtime_copy"] = str(runtime)
    # The immutable predecessor control is a verified, read-only input. We do not
    # materialize 24 redundant 118 MB copies of it in this arm's 3 GiB store.
    inputs["fields"]["candidate"] = {"npz": npz, "u8": raw}
    inputs["storage_policy"] = "all new payloads in jrx1; verified jrd1 control read-only"
    save(root / "INPUTS.json", inputs)
    save(root / "JOB.json", {"name": name, "rows": rows, "field_npz": npz,
         "field_raw": raw, "inputs": control.custody.fact(root / "INPUTS.json"),
         "source": control.custody.fact(Path(__file__)), "score_claim": False,
         "owned_runtime": owned_runtime,
         "pricer_sources": [control.custody.fact(REPO / "experiments" / p) for p in
                            ("ddm_sj1_rlc1_price.py", "ddm_jg2_tail_reencode.py")],
         "raw_status": "derived exact reconstruction identity; expansion materializes only at encode"})


def read_proposals() -> tuple[list[int], dict[int, list[dict]]]:
    if control.custody.fact(control.SAMPLE)["sha256"] != control.SAMPLE_SHA:
        raise ValueError("fixed sample drift")
    sample = load(control.SAMPLE)["pairs"]
    binding = load(control.SEARCH / "BINDING.json")
    binding_sha = control.digest(json.dumps(binding, sort_keys=True).encode())
    base = np.load(control.PD4 / "base/pose_base_move52.npy")
    pd4.bind_move52(raw=control.PD4 / "parseback/0.raw")
    pose_unit = pd4.pose_s_per_pair_unit(float(base.mean()))
    seg_unit = pd4.seg_s_per_cell()
    pool = {}
    for pair in sample:
        done = load(control.SEARCH / f"pair_{pair:03d}/DONE.json")
        if done["binding_sha"] != binding_sha:
            raise ValueError("search pair source identity differs from this generation")
        for fact in done["artifacts"]:
            checked_fact(fact)
        attempt = Path(done["completed_attempt"])
        entries = {}
        for path in (attempt / "single/search_b_0.jsonl", attempt / "cluster/cluster_0.jsonl"):
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row["pair"] != pair:
                    raise ValueError("pair crossed its isolated artifact boundary")
                row["benefit_S"] = -row["credit_d_pose"] * pose_unit - row["d_cells"] * seg_unit
                row["tokens_changed"] = len(pd4.row_edits(row))
                entries[pd4.proposal_key(row)] = row
        pool[pair] = sorted(entries.values(), key=lambda r: -r["benefit_S"])[:6]
    return sample, pool


def sheets() -> None:
    sample, pool = read_proposals()
    missing = [p for p in sample if not pool[p]]
    if missing:
        save(ROOT / "MISSING_CONTROL_PAIRS.json", {"pairs": sample, "missing": missing,
             "status": "INSTRUMENT_INCOMPLETE_NO_ADMISSIBLE_PROPOSAL", "score_claim": False})
        raise ValueError("fixed K24 contains pairs without an admissible proposal; no invented median")
    plans = []
    for k in range(max(len(v) for v in pool.values())):
        rows = [pool[p][min(k, len(pool[p])-1)] for p in sample]
        name = f"sheet{k}"
        prepare_job(name, rows)
        plans.append({"name": name, "rows": rows})
    save(ROOT / "SHEETS.json", {"pairs": sample, "plans": plans,
         "selection": "pd4 top6 by benefit; fixed-K extension retains nonpositive benefits",
         "charge": False})


def encode(name: str) -> None:
    if not name or not name.isidentifier():
        raise ValueError("job name must be an identifier")
    root = ROOT / name
    job = load(root / "JOB.json")
    checked_fact(job["inputs"])
    checked_fact(job["source"])
    checked_fact(job["field_npz"])
    verify_live(load(root / "INPUTS.json"))
    for fact in job["pricer_sources"]:
        checked_fact(fact)
    for fact in job["owned_runtime"]:
        checked_fact(fact)
    raw_path = Path(job["field_raw"]["path"])
    if raw_path != root / "bulk/field.u8":
        raise ValueError("raw expansion is outside the owned job")
    if (root / "DONE.json").exists():
        done = load(root / "DONE.json")
        checked_fact(done["encode"])
        checked_fact(done["raw_expansion_certificate"])
        result = load(Path(done["encode"]["path"]))
        for fact in list(result["outputs"].values()) + result["archives"]:
            checked_fact(fact)
        return
    reserve_encode(root)
    if not raw_path.exists():
        control.custody.check_capacity(job["field_raw"]["bytes"] + (64 << 20))
        with np.load(job["field_npz"]["path"], allow_pickle=False) as data:
            field = np.stack([data[str(p)] for p in range(600)])
        control.locked_retain(raw_path, field.tobytes())
    checked_fact(job["field_raw"])
    os.environ.update(SJ1_RLC1_ROOT=str(root), SJ1_RLC1_BULK=str(root / "bulk"),
                      SJ1_RLC1_LIVE=load(REFERENCE)["live_tree"])
    import ddm_sj1_rlc1_price as pricer
    pricer.ROOT, pricer.BULK = root, root / "bulk"
    pricer.LIVE = Path(load(REFERENCE)["live_tree"])
    if getattr(pricer, "_jrx1_encode_invoked", False):
        raise ValueError("one encode job per process; use the resumable CLI for the next job")
    pricer._jrx1_encode_invoked = True
    original_preflight = pricer.preflight

    def bounded_preflight(path, need=1 << 20):
        with (STORE / ".custody.lock").open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            if job_bytes(root) + need + (2 << 20) > ENCODE_BUDGET:
                raise ValueError("RETENTION_BLOCK: this encoder would exceed its reserved byte budget")
            control.custody.check_capacity(need)
            original_preflight(path, need)

    pricer.preflight = bounded_preflight
    result_path = root / "encode/candidate/primary/ENCODE.json"
    if result_path.exists():
        result = load(result_path)
        for fact in list(result["outputs"].values()) + result["archives"]:
            checked_fact(fact)
    else:
        print(json.dumps({"job": name, "storage_before_heavy": control.stable_storage()}), flush=True)
        control.custody.check_capacity(64 << 20)
        pricer.encode("candidate", "primary")
        result = load(result_path)
    if result["archives"][0]["sha256"] != result["archives"][1]["sha256"]:
        raise ValueError("real encoder twins disagree")
    # Automatic lossless hygiene: certify a full round-trip from retained npz,
    # then remove ONLY its redundant raw expansion. Every archive and checkpoint
    # remains. A resume rehydrates the exact field before canonical guard().
    with np.load(job["field_npz"]["path"], allow_pickle=False) as data:
        restored = np.stack([data[str(p)] for p in range(600)])
    if control.digest(restored.tobytes()) != job["field_raw"]["sha256"]:
        raise ValueError("CERTIFY_OR_BLOCK: field reconstruction does not match")
    certificate = save(root / "RAW_EXPANSION_CERTIFICATE.json", {
        "original": job["field_raw"], "retained_payload": job["field_npz"],
        "rebuild": "np.stack([np.load(field.npz)[str(p)] for p in range(600)]).tobytes()",
        "source": job["source"], "argv": sys.argv, "seed": 20260912,
        "classification": "certified redundant raw expansion; all field bytes retained losslessly in npz",
        "score_claim": False, "delete_only_after_verified_full_sha": True})
    checked_fact(job["field_raw"])
    raw_path.unlink()
    save(root / "DONE.json", {"encode": control.custody.fact(result_path),
         "archives": result["archives"], "delta_archive_bytes": result["archives"][0]["bytes"]-179332,
         "rows": job["rows"], "raw_expansion_removed": True, "score_claim": False,
         "raw_expansion_certificate": certificate,
         "carrier_boundary": "shipped carrier held; resolved-pose codes not embodied in pricer archive"})


def winners() -> None:
    sheets_doc = load(ROOT / "SHEETS.json")
    if control.custody.fact(TAIL_CONTROL)["sha256"] != TAIL_CONTROL_SHA:
        raise ValueError("predecessor control ledger receipt drift")
    base = load(TAIL_CONTROL)
    for fact in base["archives"]:
        checked_fact(fact)
    candidates = {p: {} for p in sheets_doc["pairs"]}
    repeated = []
    for plan in sheets_doc["plans"]:
        done = load(ROOT / plan["name"] / "DONE.json")
        checked_fact(done["encode"])
        priced = load(Path(done["encode"]["path"]))
        for row in plan["rows"]:
            p = row["pair"]
            bits = priced["per_frame_bits"][p] - base["per_frame_bits"][p]
            key = pd4.proposal_key(row)
            if key in candidates[p]:
                repeated.append({"pair": p, "proposal": key, "sheet": plan["name"],
                                 "first_bits": candidates[p][key]["ranking_bits"], "repeat_bits": bits})
            else:
                candidates[p][key] = dict(row, ranking_bits=bits)
    selected = []
    for p, rows in candidates.items():
        winner = max(rows.values(), key=lambda r: (pd4.credit_per_bit(r["benefit_S"], r["ranking_bits"]), r["benefit_S"]))
        name = f"pair{p:03d}"
        prepare_job(name, [winner])
        selected.append({"name": name, "row": winner})
    save(ROOT / "WINNERS.json", {"pairs": sheets_doc["pairs"], "jobs": selected,
         "repeated_price_diagnostics_not_selected": repeated,
         "ranker": "pd4 credit_per_bit on actual sheet ledger; individual whole archive is the charge"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sheets", "encode", "winners"))
    parser.add_argument("--name")
    parser.add_argument("--resume-from", type=Path, required=True)
    args = parser.parse_args()
    if args.resume_from.resolve() != STORE.resolve():
        raise ValueError("wrong owned resume store")
    if os.getpriority(os.PRIO_PROCESS, 0) != 0:
        raise ValueError("charter requires niceness zero")
    {"sheets": sheets, "encode": lambda: encode(args.name), "winners": winners}[args.stage]()


if __name__ == "__main__":
    main()
