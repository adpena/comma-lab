#!/usr/bin/env bash
# Run the unchanged TC1 public proof against CMP1's own root; no render/scorer.
# Launch only after both encode workers exit and the final archive is staged.
set -euo pipefail
cd /Users/adpena/Projects/pact
root=/Volumes/VertigoDataTier/pact/ddm_cmp1_compose
frame() {
  .venv/bin/python -B -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1])/"public_identity/frame_checkpoints/LATEST.json"; print(json.loads(p.read_text())["frame"] if p.exists() else 0)' "$root"
}
for attempt in 1 2 3 4 5 6 7 8; do
  before="$(frame)"
  df -h /Volumes/VertigoDataTier /Volumes/APDataStore
  set +e
  .venv/bin/python tools/safe_run.py --timeout 780 --rss-mb 4096 --label "cmp1_public_${attempt}" -- \
    .venv/bin/python -B experiments/ddm_tc1_public_proof.py decode --root "$root" --stop 600
  code="$?"
  set -e
  if [[ "$code" == 0 ]]; then
    .venv/bin/python -B -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1])/"public_identity/PUBLIC_FIELD_IDENTITY.json"; r=json.loads(p.read_text()); assert r["pairs"]==600 and r["exact_field_identity"]' "$root"
    exit 0
  fi
  after="$(frame)"
  if [[ "$code" != 124 || "$after" -le "$before" ]]; then
    echo "Refused public continuation: code=$code before=$before after=$after" >&2
    exit "$code"
  fi
  echo "Resuming public decoder after bounded timeout: $before -> $after"
done
echo "Exhausted eight bounded public stages; retained checkpoint requires inspection" >&2
exit 1
