#!/usr/bin/env bash
# Resume a single independent encoder only after its previous watchdog exited.
# Invoked through tools/launch_detached_process.py; never starts a scorer.
set -euo pipefail
cd /Users/adpena/Projects/pact
tag="$1"
case "$tag" in primary|repeat) ;; *) exit 2 ;; esac
root=/Volumes/VertigoDataTier/pact/ddm_cmp1_compose
frame() {
  .venv/bin/python -B -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1])/"encode"/sys.argv[2]/"LATEST.json"; print(json.loads(p.read_text())["frame"])' "$root" "$tag"
}
for attempt in 1 2 3 4 5 6 7 8; do
  before="$(frame)"
  df -h /Volumes/VertigoDataTier /Volumes/APDataStore
  set +e
  .venv/bin/python tools/safe_run.py --timeout 780 --rss-mb 4096 --label "cmp1_${tag}_resume_${attempt}" -- \
    .venv/bin/python -B experiments/ddm_cmp1_compose.py encode --tag "$tag" --stop 600 --resume-from "$root"
  code="$?"
  set -e
  if [[ "$code" == 0 ]]; then
    exit 0
  fi
  after="$(frame)"
  if [[ "$code" != 124 || "$after" -le "$before" ]]; then
    echo "Refused automatic continuation: code=$code before=$before after=$after" >&2
    exit "$code"
  fi
  echo "Resuming source-bound $tag after bounded timeout: $before -> $after"
done
echo "Exhausted eight bounded stages; retained checkpoint requires inspection" >&2
exit 1
