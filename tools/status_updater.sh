#!/bin/bash
cd "$(git rev-parse --show-toplevel)"
while true; do
    .venv/bin/python -c "
import json, os, subprocess
from datetime import datetime, timezone, timedelta
weights_dir = 'experiments/postfilter_weights'
best = {'scorer': 999, 'epoch': 0}
for f in os.listdir(weights_dir):
    if f.endswith('_best_meta.json'):
        try:
            d = json.load(open(os.path.join(weights_dir, f)))
            if d.get('scorer', 999) < best['scorer']:
                best = d
                best['tag'] = f.replace('postfilter_','').replace('_best_meta.json','')
        except: pass
trainers = int(subprocess.run('ps aux | grep python | grep train_postfilter | grep -v grep | wc -l', shell=True, capture_output=True, text=True).stdout.strip() or '0')
deadline = datetime(2026, 5, 4, 11, 59, tzinfo=timezone.utc)
days = (deadline - datetime.now(timezone.utc)).total_seconds() / 86400
status = {'score':1.727,'rank':1,'lead':0.163,'days_left':round(days,1),'best_training_scorer':round(best['scorer'],4),'best_training_epoch':best.get('epoch',0),'gap_to_proxy':round(best['scorer']-3.55,3),'trainers_alive':trainers,'updated':datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M CT')}
json.dump(status, open('reports/graphs/site/status.json','w'), indent=2)
print(f'Updated: {best[\"scorer\"]:.4f} ep{best.get(\"epoch\",0)} {trainers}t {days:.1f}d')
" 2>/dev/null
    cd reports/graphs/site && wrangler pages deploy . --project-name comma-lab > /dev/null 2>&1
    cd "$(git rev-parse --show-toplevel)"
    sleep 600
done
