#!/usr/bin/env bash
# Run ONE bounded, resumable 600-frame encode for ddm_sj1's pass-5 pricing.
#
# Each encode is a single-threaded ~30-minute traversal that checkpoints every ten
# frames.  A bounded watchdog plus a retry loop is the shape cmp1 used and the reason
# a hibernate or an OOM costs at most one stage rather than the whole run; the loop
# refuses to continue on any exit that is not a clean watchdog timeout that made
# forward progress, so a real fault surfaces instead of being retried forever.
#
# argv: <field: control|candidate> <tag>
set -euo pipefail
cd /Users/adpena/Projects/pact

field="${1:?field}"
tag="${2:?tag}"
case "$field" in control|candidate) ;; *) echo "bad field" >&2; exit 2 ;; esac

root=/Volumes/VertigoDataTier/pact/ddm_sj1_pass5_price
work="$root/encode/$field/$tag"

frame() {
  if [ -f "$work/LATEST.json" ]; then
    .venv/bin/python -B -c 'import json,sys;print(json.loads(open(sys.argv[1]).read())["frame"])' "$work/LATEST.json"
  else
    echo 0
  fi
}

for attempt in $(seq 1 20); do
  if [ -f "$work/ENCODE_0600.json" ]; then
    echo "encode already complete: $field/$tag"
    exit 0
  fi
  before="$(frame)"
  df -h /Volumes/VertigoDataTier /Volumes/APDataStore
  set +e
  nice -n 10 .venv/bin/python tools/safe_run.py --timeout 1800 --rss-mb 6144 \
    --label "sj1p5_${field}_${tag}_${attempt}" -- \
    .venv/bin/python -B experiments/ddm_sj1_pass5_price.py encode --field "$field" --tag "$tag" --stop 600
  code="$?"
  set -e
  if [ "$code" = 0 ]; then
    echo "encode complete: $field/$tag"
    exit 0
  fi
  after="$(frame)"
  if [ "$code" != 124 ] || [ "$after" -le "$before" ]; then
    echo "Refused automatic continuation: code=$code before=$before after=$after" >&2
    exit "$code"
  fi
  echo "Resuming $field/$tag after bounded timeout: $before -> $after"
done
echo "Exhausted twenty bounded stages; the retained checkpoint needs inspection" >&2
exit 1
