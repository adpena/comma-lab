"""Bounded ddm_cd3 custody preparation; never execute or mutate SSD sources.

Consumes the retained sw2 census and cd3 per-source review. Reproductions and
inert source copies are kept on disk. This is a small source/metadata operation,
not a bulk materializer; a 2 MiB input cap fails closed before any write.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from experiments.ddm_cs1_disposition_ssd_code import SHA_LINE, SIZE_LINE, reproduce
from experiments.ddm_sw1_disposition_ssd_code import append_rows
from tools.audit_ssd_authored_signal import reachable_git_blobs

WORK = Path(".omx/research/ddm_cd3_20260910")
LEDGER = Path(".omx/research/ddm_cs1_ssd_code_certify_20260909.jsonl")
MEMOS = {
    "ddm_g8r": ".omx/research/ddm_g8r_compress_adversarial_review_20260902.md",
    "ddm_gt2": ".omx/research/ddm_gt2_dali_gt_selector_repoint_20260820.md",
    "ddm_ql1": ".omx/research/ddm_ql1_retired_lineage_test_quarantine_20260903.md",
    "ddm_ps1": ".omx/research/ddm_ps1_pr140_update_packet_prep_20260904.md",
    "ddm_ps2": ".omx/research/ddm_ps2_pr140_update_packet_stage7_20260904.md",
    "ddm_gb2": ".omx/research/ddm_gb2_generated_basis_unconditional_lattice_bound_20260909.md",
    "ddm_sm1": ".omx/research/ddm_sm1_semantic_section_shared_mixer_coder_priced_closed_form_20260909.md",
    "ddm_cmp1": ".omx/research/ddm_cmp1_compose_rc3_tc1_20260909.md",
    "ddm_bnd1": ".omx/research/ddm_bnd1_boundary_representation_closed_form_pricing_20260909.md",
    "ddm_cmp2": ".omx/research/ddm_cmp2_compose_sm1_fe1_20260909.md",
    "ddm_rp1": ".omx/research/ddm_rp1_rate_directed_token_predistortion_20260909.md",
    "ddm_sj1": ".omx/research/ddm_sj1_multipass_token_predistortion_20260905.md",
}
LITERAL_FIXTURES = {
    "34a2a07a1c9185a9a50280ac314755ca7fc453c6": b"raise AssertionError\n",
    "51fa5d94e1a1ee70ef71f330d8891cfa39ffbe55": b"corrupt cached source",
    "4287ca8617970fa8fc025b75cb319c7032706910": b"#",
}
SMALL_FIXTURES = {
    "dd0aaee4ceaa1c374b4eb81c7115dd5ab4a0faf7": "setup/XPASS control explained in owner memo QUARANTINE MECHANISM",
    "15d2e243935a75f73eee654fec88e9f3ed0bce7c": "338-byte undeclared-GT control hash pinned in owner memo",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def put(path: Path, data: bytes) -> None:
    """Resume only from byte-identical output; never overwrite a differing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"output drift: {path}")
    else:
        temporary = path.with_suffix(".pending")
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def owner_evidence(owner: str) -> tuple[str, str]:
    memo = MEMOS[owner]
    commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", memo], text=True).strip()
    if len(commit) != 40 or not Path(memo).is_file():
        raise ValueError(f"missing committed owner memo: {owner}: {memo}")
    return memo, commit


def prepare() -> list[dict]:
    checks = json.loads((WORK / "source_checks.json").read_text())
    eligible = {r["blob_sha1"]: r for r in json.loads((WORK / "eligible_source_review.json").read_text())}
    repins = {r["blob_sha1"]: r for r in json.loads((WORK / "repin_proofs_plan.json").read_text())}
    retained = {
        r["blob"]: r
        for r in json.loads(Path(".omx/research/ddm_sw1_20260910/ssd_disposition_summary.json").read_text())["retained"]
    }
    if len(checks) != 41 or len({r["blob_sha1"] for r in checks}) != 41:
        raise ValueError("sw2 baseline is not 41 distinct blobs")
    if sum(r["size_bytes"] for r in checks) > 2 * 1024 * 1024:
        raise ValueError("bounded source custody cap exceeded")
    reachable = reachable_git_blobs()
    rows = []
    for item in checks:
        source = Path(item["representative_path"])
        data = source.read_bytes()
        sha = item["blob_sha1"]
        if blob(data) != sha or digest(data) != item["sha256"] or len(data) != item["size_bytes"]:
            raise ValueError(f"source drift: {source}")
        owner = "_".join(source.parts[4].split("_")[:2])
        memo, memo_commit = owner_evidence(owner)
        row = {
            "schema": "ddm_cd3.custody.v1", "event_type": "source_disposition",
            "date": "2026-09-10", "axis": "macOS-CPU filesystem/custody apparatus",
            "score_claim": False, "source_blob_sha1": sha, "source_sha256": digest(data),
            "source_path": str(source), "source_bytes": len(data),
            "instance_count": item["instance_count"], "owner": owner,
            "owner_memo": memo, "owner_memo_commit": memo_commit,
            "consumer_store": str(LEDGER), "disposition": "RETAIN_WITH_OWNER",
            "follow_on_disposition": "QUEUED-WITH-A-FIRE-ORDER",
        }
        if sha in repins:
            template = repins[sha]["template"]
            if template["blob_sha1"] not in reachable:
                raise ValueError("reproducer template lost reachability")
            original = subprocess.check_output(["git", "cat-file", "blob", template["blob_sha1"]])
            rebuilt = reproduce(original, data)
            if rebuilt != data:
                raise ValueError("archive-pin reconstruction failed")
            target = WORK / "reproduced" / f"{sha}.blob"
            put(target, rebuilt)
            row.update(
                disposition="CERTIFIED_REPRODUCED", follow_on_disposition="FOLDED",
                reason="Exact reconstruction from a reachable template with only the two archive-pin lines replaced.",
                reproducer={"template": template, "sha_line": SHA_LINE.findall(data)[0].decode(),
                            "size_line": SIZE_LINE.findall(data)[0].decode(),
                            "helper": "experiments/ddm_cs1_disposition_ssd_code.py::reproduce"},
                retained_reproduction=str(target),
                fire_trigger="Certificate proof is complete; this certificate does not authorize SSD cleanup.",
            )
        elif sha in LITERAL_FIXTURES or sha in SMALL_FIXTURES:
            literal = LITERAL_FIXTURES.get(sha, data)
            if literal != data:
                raise ValueError("literal fixture mismatch")
            if sha in SMALL_FIXTURES and len(data) > 407:
                raise ValueError("bounded literal fixture exceeded its reviewed size")
            if sha.startswith("15d2") and digest(data) not in Path(memo).read_text():
                raise ValueError("GT control no longer matches owning memo")
            target = WORK / "reproduced" / f"{sha}.blob"
            put(target, literal)
            row.update(
                disposition="CERTIFIED_REPRODUCED", follow_on_disposition="FOLDED",
                reason="Deliberate bounded detector fixture, not operational source: "
                       + SMALL_FIXTURES.get(sha, "g8r extra-source/corrupt-source/hash-tamper control"),
                reproducer={"literal_utf8": literal.decode(), "owner_memo": memo},
                retained_reproduction=str(target),
                fire_trigger="Exact fixture literal retained; no source promotion or SSD cleanup authorized.",
            )
        elif sha in eligible:
            target = WORK / "recovered_sources" / f"{sha}.blob"
            put(target, data)
            row.update(
                disposition="EXACT_SOURCE_READY_FOR_SERIALIZER", repo_relative_destination=str(target),
                reason="Real historical source/snapshot with exact owner memo; preserved inertly, never executed or adopted.",
                review_pass_1=eligible[sha]["review_pass_1"], review_pass_2=eligible[sha]["review_pass_2"],
                ruff_codes=eligible[sha]["ruff_codes"],
                fire_trigger="MAIN lands the exact source batch after manifest hash verification and checks ref reachability.",
            )
        else:
            reason = retained.get(sha, {}).get("reason", "Protected rp1 runtime source version; owner landing required.")
            if "correctness-risk" in reason:
                trigger = f"{owner} supplies a reviewed exact source/reproducer resolving {source.name}'s F821/B023/syntax findings; then record its hash and reachable commit."
            elif "control" in reason:
                trigger = f"{owner} supplies a committed generator and exact reproduction for {source.name} ({sha[:12]}); verify equality before certifying."
            else:
                trigger = f"At this handoff's harvest, {owner} reconciles {source.name} ({sha[:12]}) against its completed packet/source landing and records an exact reachable blob or a hash-bound reproducer; future harvest is not presumed."
            row.update(reason=reason, fire_trigger=trigger)
        rows.append(row)
    if len(retained) != 20 or not set(retained).issubset({r["source_blob_sha1"] for r in rows}):
        raise ValueError("original retained-owner denominator drift")
    put(WORK / "dispositions.jsonl", "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode())
    owner_rows = []
    for r in rows:
        if r["source_blob_sha1"] in retained:
            entry = dict(r, event_type="retain_owner_fire_order")
            entry["supersedes_schema"] = "ddm_sw2.retain_owner_fire_order.v1"
            owner_rows.append(entry)
    put(WORK / "owner_fire_orders.jsonl", "".join(json.dumps(r, sort_keys=True) + "\n" for r in owner_rows).encode())
    return rows


def append_prepared_rows() -> None:
    """Append only missing exact events; a changed existing cd3 event is an error."""
    existing = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    index = {
        (r["event_type"], r["source_blob_sha1"]): r
        for r in existing if r.get("schema") == "ddm_cd3.custody.v1"
    }
    additions = []
    for name in ("dispositions.jsonl", "owner_fire_orders.jsonl"):
        for line in (WORK / name).read_text().splitlines():
            row = json.loads(line)
            key = (row["event_type"], row["source_blob_sha1"])
            if key in index and index[key] != row:
                raise ValueError(f"existing disposition differs: {key}")
            if key not in index:
                additions.append(row)
    append_rows(LEDGER, additions)


if __name__ == "__main__":
    prepare()
    append_prepared_rows()
