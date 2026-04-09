#!/bin/bash
# Fetch latest competition PR scores from GitHub
# Output: JSON array of {name, score, date}
# Cache: only fetches if cache is older than 10 minutes
CACHE="/tmp/comma_lab_leaderboard.json"
AGE=$(($(date +%s) - $(stat -f%m "$CACHE" 2>/dev/null || echo 0)))

if [ "$AGE" -gt 600 ] || [ ! -f "$CACHE" ]; then
  gh api "repos/commaai/comma_video_compression_challenge/pulls?state=all&per_page=50" \
    --jq '[.[] | {title: .title, date: .created_at}]' 2>/dev/null | \
  python3 -c "
import sys, json, re
prs = json.load(sys.stdin)
scores = []
seen = set()
for pr in prs:
    m = re.search(r'score:\s*([\d.]+)', pr['title'])
    name = pr['title'].split('(')[0].strip().split('submission')[0].strip()
    if m and name not in seen:
        scores.append({'name': name, 'score': float(m.group(1)), 'date': pr['date'][:10]})
        seen.add(name)
scores.sort(key=lambda x: x['score'])
json.dump(scores[:10], sys.stdout, indent=2)
" > "$CACHE" 2>/dev/null
fi
cat "$CACHE" 2>/dev/null || echo "[]"
