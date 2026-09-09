"""Reproduce archive-pin-only build products; preserve all other debt as blocked.

No SSD writes and no execution of recovered sources. A certificate requires exact
byte reconstruction from a reachable Git blob, not similarity or a filename guess.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

SHA_LINE = re.compile(rb"(?m)^ARCHIVE_SHA256 = [\'\"][0-9a-f]{64}[\'\"]$")
SIZE_LINE = re.compile(rb"(?m)^ARCHIVE_BYTES = [0-9]+(?:_[0-9]+)*$")


def normalize(data):
    return SIZE_LINE.sub(b"ARCHIVE_BYTES = SIZE", SHA_LINE.sub(b'ARCHIVE_SHA256 = "PIN"', data))


def reproduce(template, target):
    sha_lines = SHA_LINE.findall(target)
    size_lines = SIZE_LINE.findall(target)
    if len(sha_lines) != 1 or len(size_lines) != 1:
        return None
    if len(SHA_LINE.findall(template)) != 1 or len(SIZE_LINE.findall(template)) != 1:
        return None
    rebuilt = SIZE_LINE.sub(lambda _: size_lines[0], SHA_LINE.sub(lambda _: sha_lines[0], template))
    return rebuilt if rebuilt == target else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--apply-certificates", action="store_true")
    args = parser.parse_args()
    for path in (args.work, args.ledger):
        if Path("/Volumes") in path.resolve().parents:
            raise ValueError("SSD mutation is forbidden")
    spec = importlib.util.spec_from_file_location("ssd_audit", "tools/audit_ssd_authored_signal.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    reachable = audit.reachable_git_blobs()
    templates = json.loads((args.work / "repin_templates.json").read_text())
    keeper = json.loads((args.work / "keeper_before.json").read_text())
    landings = {r["source"]: r for r in json.loads((args.work / "source_landings.json").read_text())}
    existing = audit.load_certified()
    report = json.loads(args.scan.read_text())
    rows = []
    retained = args.work / "reproduced_code"
    retained.mkdir(exist_ok=True)
    for item in report["owed"]:
        if item["ext"].lstrip(".") not in ("py", "sh"):
            continue
        source = Path(item["representative_path"])
        data = source.read_bytes()
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        if blob != item["blob_sha1"]:
            raise ValueError(f"source changed since scan: {source}")
        sha = hashlib.sha256(data).hexdigest()
        row = {
            "path": str(source),
            "blob_sha1": blob,
            "sha256": sha,
            "bytes": len(data),
            "instance_count": item["instance_count"],
            "owner": "MAIN / source-owner reconciliation",
            "disposition": "BLOCKED",
            "reason": "No verified committed reproducer or terminal keeper owner for this exact authored source; bytes retained.",
            "consumer_store": str(args.ledger),
            "fire_trigger": "Owner supplies exact reproducer or keeper-terminal source review and serializer landing.",
        }
        arm_directory = source.parts[4] if len(source.parts) > 4 else ""
        owners = [r for name, r in keeper.items() if name == arm_directory]
        if owners and any(r.get("status") not in ("landed", "closed") for r in owners):
            row["reason"] = "Keeper owner is not landed/closed; charter forbids landing live-arm code."
            row["owner"] = owners[0]["name"]
        if blob in reachable:
            row.update(disposition="MATCHED", reason="Exact blob now reachable from a Git ref; no SSD change.")
        elif str(source) in landings:
            row.update(
                disposition=landings[str(source)]["disposition"],
                landing=landings[str(source)],
                owner="ddm_cs1",
                reason="Keeper-closed S1A shell driver copied byte-identically; syntax and source review complete.",
            )
        elif source.name == "inflate.py":
            template = templates.get(hashlib.sha256(normalize(data)).hexdigest())
            if template and template["blob_sha1"] in reachable:
                original = subprocess.check_output(["git", "cat-file", "blob", template["blob_sha1"]])
                rebuilt = reproduce(original, data)
                if rebuilt is not None:
                    output = retained / (sha + ".bin")
                    output.write_bytes(rebuilt)
                    row.update(
                        disposition="CERTIFIED",
                        owner="ddm_cs1",
                        reason="Generated archive-pin-only runtime copy; exact reconstruction from reachable template passes.",
                        reproducer={
                            "git_blob_sha1": template["blob_sha1"],
                            "source_path": template["path"],
                            "operation": "replace exactly one ARCHIVE_SHA256 line and one ARCHIVE_BYTES line",
                            "sha_line": SHA_LINE.findall(data)[0].decode(),
                            "size_line": SIZE_LINE.findall(data)[0].decode(),
                            "helper": "experiments/ddm_cs1_disposition_ssd_code.py::reproduce",
                        },
                        retained_reproduction=str(output),
                    )
        elif str(source).endswith("/retained_controls/ddm_sw1_absolute_path_positive_control.py"):
            literal = b'ROOT = "/Volumes/PrivateSw1/control"\n'
            if data == literal:
                output = retained / (sha + ".bin")
                output.write_bytes(literal)
                row.update(
                    disposition="CERTIFIED",
                    owner="ddm_cs1",
                    reason="Historical deliberate absolute-path detector fixture, proved by ddm_sw1_portable_paths_secrets_scrub_20260820.md table P3.",
                    reproducer={"literal_utf8": literal.decode()},
                    retained_reproduction=str(output),
                )
        rows.append(row)
    expected = sum(Path(item["representative_path"]).suffix in (".py", ".sh") for item in report["owed"])
    if len(rows) != expected:
        raise ValueError(f"code census disagreement: {len(rows)} dispositions for {expected} source paths")
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    if args.apply_certificates:
        for row in rows:
            if row["disposition"] == "CERTIFIED" and row["blob_sha1"] not in existing:
                audit.append_certification(
                    row["blob_sha1"],
                    row["reason"] + " Reproducer and sha256: " + str(args.ledger),
                    "ddm_cs1",
                    row["path"],
                )
    print(
        json.dumps(
            {
                status: sum(r["disposition"] == status for r in rows)
                for status in sorted({r["disposition"] for r in rows})
            }
        )
    )


if __name__ == "__main__":
    main()
