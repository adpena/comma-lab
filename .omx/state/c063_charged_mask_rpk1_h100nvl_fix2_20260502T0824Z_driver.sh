#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
cd "$WORKSPACE"

OUT="experiments/results/vast_live_harvest/c063_charged_mask_rpk1_h100nvl_fix2_20260502T0824Z"
ARCHIVE_SRC="$OUT/archive.zip"
mkdir -p "$OUT"
log() { echo "[c063-charged-rpk1-h100nvl-fix2] $(date -u +%FT%TZ) $*" | tee -a "$OUT/driver.log"; }

cat > "$OUT/source_sha256s.expected" <<'SHA'
submissions/robust_current/inflate.sh=86449a1f52ac6b2be120d47287b8410f915dce7e562c69f480103f6e527c6017
submissions/robust_current/inflate_renderer.py=697d1d4bdbfa224506ab88309163f51a0752c356fe4b8848c6c42985d5afa1da
submissions/robust_current/inflate_renderer_grayscale.py=383407e20aff8a5b7791f61d3cdfd7516ef870bab99145854ff2f399471b8f69
submissions/robust_current/unpack_renderer_payload.py=cac8cde654f2d875d4567c18b77d573af91c29dbb0b05b7934dc7e019ae66f49
experiments/contest_auth_eval.py=8d9dd3e1e9f97245623c938aa9b28a41fdd3e41752208b16b218131906f7ef53
scripts/remote_archive_only_eval.sh=8d1069de4f2426108beb44e519f83428608c1354ea2752b426cddd1f5ab815f8
SHA
python3 - <<'PY'
from pathlib import Path
import hashlib

bad = []
for line in Path("experiments/results/vast_live_harvest/c063_charged_mask_rpk1_h100nvl_fix2_20260502T0824Z/source_sha256s.expected").read_text().splitlines():
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

python3 - <<'PY'
from pathlib import Path
import hashlib
import json
import zipfile

archive = Path("experiments/results/vast_live_harvest/c063_charged_mask_rpk1_h100nvl_fix2_20260502T0824Z/archive.zip")
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
    "candidate_family": "packed_rpk1_charged_amr1_budget4096_over_crf52_mask_base",
}
Path("experiments/results/vast_live_harvest/c063_charged_mask_rpk1_h100nvl_fix2_20260502T0824Z/archive_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY

log "exact CUDA diagnostic eval"
export ARCHIVE_PATH="$WORKSPACE/$ARCHIVE_SRC"
export ARCHIVE_LABEL="archive_eval_c063_charged_mask_rpk1_budget4096_h100nvl_fix2_20260502"
export LOG_DIR="$WORKSPACE/$OUT/exact_eval"
export PREDICTED_LOW="0.18"
export PREDICTED_HIGH="2.35"
export CONTROLLED_BASELINE="C-063 frontier with CRF52 mask base plus charged RPK1/AMR1 budget4096 repair; H100 diagnostic only"
export REQUIRED_SOURCE_SHA256S="$(cat "$OUT/source_sha256s.expected")"
bash scripts/remote_archive_only_eval.sh
log "done"
