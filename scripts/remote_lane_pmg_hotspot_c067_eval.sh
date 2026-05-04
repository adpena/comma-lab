#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
cd "$WORKSPACE"

if [[ "${ALLOW_REPLAY_EXACT_NEGATIVE_PMG:-0}" != "1" ]]; then
  echo "PMG_EXACT_NEGATIVE_REPLAY_GUARD: PMG-HOTSPOT C067 exact eval is preserved for forensics only; set ALLOW_REPLAY_EXACT_NEGATIVE_PMG=1 for an intentional replay."
  exit 88
fi

OUT="experiments/results/vast_live_harvest/pmg_hotspot_c067_h100_20260502T1350Z"
CANDIDATE_DIR="experiments/results/pmg_hotspot_candidate_c067_20260502"
ARCHIVE_SRC="$CANDIDATE_DIR/archive.zip"
mkdir -p "$OUT"
log() { echo "[pmg-hotspot-c067-h100] $(date -u +%FT%TZ) $*" | tee -a "$OUT/driver.log"; }

cat > "$OUT/source_sha256s.expected" <<'SHA'
submissions/robust_current/inflate.sh=86449a1f52ac6b2be120d47287b8410f915dce7e562c69f480103f6e527c6017
submissions/robust_current/inflate_renderer.py=ba334748b365fbf0055323177fb21842528dbe37a2b0ef5a303260da00a09f40
submissions/robust_current/unpack_renderer_payload.py=3c31a4fbf5c11956ab9a301e735f6b3a5f7bc5034bddd67da0f956c04f0cd1a4
experiments/contest_auth_eval.py=7ef32b5bdcb1043898cb9d7913cec354f96cbfa13bceed75a5cfbef2e127655c
scripts/remote_archive_only_eval.sh=c457b819749186bd5d3bdfac139fb29d664b6071aba69f8c435b2554b80333f1
experiments/results/pmg_hotspot_candidate_c067_20260502/archive.zip=3ab4d0c85ae15325b61b7838c375383a460e7ad1b911b677101fda50e2611c53
experiments/results/pmg_hotspot_candidate_c067_20260502/build_manifest.json=4195dfe589eec93de0d0805cf0cd380ef2165baf397f1d303f3d9bde2c54b910
SHA

python3 - <<'PY'
from pathlib import Path
import hashlib

bad = []
for line in Path("experiments/results/vast_live_harvest/pmg_hotspot_c067_h100_20260502T1350Z/source_sha256s.expected").read_text().splitlines():
    rel, expected = line.split("=", 1)
    actual = hashlib.sha256(Path(rel).read_bytes()).hexdigest()
    if actual != expected:
        bad.append((rel, expected, actual))
if bad:
    for rel, expected, actual in bad:
        print(f"SHA_MISMATCH {rel} expected={expected} actual={actual}")
    raise SystemExit(10)
print("SOURCE_SHA_PREFLIGHT_OK")
PY

cp "$ARCHIVE_SRC" "$OUT/archive.zip"
cp "$CANDIDATE_DIR/build_manifest.json" "$OUT/build_manifest.json"
cp "$CANDIDATE_DIR/archive_byte_accounting.json" "$OUT/archive_byte_accounting.json" 2>/dev/null || true

python3 - <<'PY'
from pathlib import Path
import hashlib
import json
import zipfile

archive = Path("experiments/results/vast_live_harvest/pmg_hotspot_c067_h100_20260502T1350Z/archive.zip")
with zipfile.ZipFile(archive) as zf:
    members = [
        {
            "name": info.filename,
            "size_bytes": info.file_size,
            "compressed_size_bytes": info.compress_size,
            "crc": f"{info.CRC:08x}",
        }
        for info in zf.infolist()
    ]
payload = {
    "archive_bytes": archive.stat().st_size,
    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "members": members,
    "score_claim": False,
    "promotion_eligible": False,
    "candidate_family": "pmg_hotspot_cmg3_residual_over_c067",
}
Path("experiments/results/vast_live_harvest/pmg_hotspot_c067_h100_20260502T1350Z/archive_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY

log "exact CUDA diagnostic eval"
export ARCHIVE_PATH="$WORKSPACE/$OUT/archive.zip"
export ARCHIVE_LABEL="archive_eval_pmg_hotspot_c067_h100_20260502"
export LOG_DIR="$WORKSPACE/$OUT/exact_eval"
export PREDICTED_LOW="0.18"
export PREDICTED_HIGH="2.50"
export CONTROLLED_BASELINE="C067 A++ frontier with PMG-HOTSPOT CMG3 residual mask self-compression; H100 diagnostic only"
export REQUIRED_SOURCE_SHA256S="$(cat "$OUT/source_sha256s.expected")"
bash scripts/remote_archive_only_eval.sh
log "done"
