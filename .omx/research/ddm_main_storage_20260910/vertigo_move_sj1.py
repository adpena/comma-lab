import hashlib, json, os, shutil, sys, time
from pathlib import Path
SRC_ROOTS = [Path("/Volumes/VertigoDataTier/pact/ddm_sj1_pass5_price"), Path("/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price"), Path("/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion")]
DST_ROOT = Path("/Volumes/APDataStore/pact/cold_store_sj1_bulk_20260910")
KEEP_EXT = {".json", ".jsonl", ".md", ".log", ".txt", ".npy", ".npz", ".zip", ".py", ".sh", ".bin", ".pid"}
MIN = 512 * 1024 * 1024
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""): h.update(c)
    return h.hexdigest()
moved = 0; log = []
for root in SRC_ROOTS:
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() in KEEP_EXT or p.stat().st_size < MIN: continue
        if "SEAL" in p.name or "archive" in p.name or "candidate_runtime" in str(p): continue
        rel = p.relative_to("/Volumes/VertigoDataTier/pact"); dst = DST_ROOT / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        s0 = sha(p); shutil.copy2(p, dst); s1 = sha(dst)
        if s1 != s0: dst.unlink(); log.append({"path": str(p), "status": "COPY_SHA_MISMATCH"}); continue
        man = {"moved_to": str(dst), "sha256": s0, "bytes": p.stat().st_size, "reason": "Vertigo under the 40 GiB serializer reserve; rebuildable bulk of a finished arm cold-stored, bytes preserved", "rebuildable_from": "the arm's sealed archive + candidate_runtime via its parse-back/render stages (see the arm memo)", "moved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        Path(str(p) + ".MOVED.json").write_text(json.dumps(man, indent=1)); os.unlink(p); moved += man["bytes"]; log.append({"path": str(p), "status": "MOVED", "bytes": man["bytes"], "sha256": s0})
        print(json.dumps(log[-1]), flush=True)
Path("/Volumes/APDataStore/pact/cold_store_sj1_bulk_20260910/MOVE_LOG.jsonl").write_text("\n".join(json.dumps(r) for r in log) + "\n")
print(json.dumps({"moved_bytes": moved, "moved_gib": round(moved / 2**30, 2), "rows": len(log)}))
