#!/bin/sh
set -eu
cd /Users/adpena/Projects/pact
exec .venv/bin/python - <<'PY'
import datetime,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,'experiments')
import ddm_psa2_rgb_bias as p
out=Path('.omx/research/ddm_psa2_20260916')
print(json.dumps({'storage_before_heavy':p.capacity(4<<20)}),flush=True)
checks=0
for pair in p.load(p.STORE/'SAMPLE.json')['pairs']:
    root=p.STORE/'pairs'/f'pair_{pair:03d}'
    for name in ['BANK.json','CONTROL.json']:
        for artifact in p.load(root/name)['artifacts']:
            p.checked(artifact);checks+=1
    if checks%40==0:print(json.dumps({'verified_artifact_references':checks}),flush=True)
for price in (p.STORE/'prices').glob('*/PRICE.json'):
    for artifact in p.load(price)['artifacts']:
        p.checked(artifact);checks+=1
for pair in p.load(p.STORE/'SAMPLE.json')['pairs']:
    for name in ['SEARCH.json','RESOLVED.json']:
        path=p.STORE/'pairs'/f'pair_{pair:03d}'/name
        if path.exists():p.checked(p.load(path)['selected']);checks+=1
binding=p.load(p.STORE/'BINDING.json')
for entry in binding['runtime']:
    p.checked(entry['source']);p.checked(entry['copy']);checks+=2
old=binding['producer']
retained=p.fact(p.STORE/'sources/prepare_producer.py')
assert old['sha256']==retained['sha256'] and old['bytes']==retained['bytes']
old_chain=p.load(p.STORE/'FOLLOWTHROUGH_BINDING.json')['sources']
old_pose=next(x for x in old_chain if x['path'].endswith('ddm_psa2_resolve_price.py'))
old_pose_copy=p.fact(p.STORE/'sources/resolve_price_producer.py')
assert old_pose['sha256']==old_pose_copy['sha256'] and old_pose['bytes']==old_pose_copy['bytes']
source_revisions={'prepare_original_fact':old,'prepare_retained_snapshot':retained,
 'superseded_chain_pins':old_chain,'superseded_pose_source_retained':old_pose_copy,
 'active_sources':p.load(p.STORE/'FOLLOWTHROUGH_V2_BINDING.json')['sources'],
 'census_source':p.fact(p.REPO/'experiments/ddm_psa2_census.py'),
 'resume_source':p.fact(p.REPO/'experiments/ddm_psa2_resume.py'),
 'note':'RGB search and original waiter source unchanged. Original waiter must refuse the old pose pin; V2 then continues under corrected pose pins. Prepare execution source retained separately.'}
for fact in source_revisions['active_sources']:p.checked(fact)
(out/'SOURCE_REVISIONS.json').write_text(json.dumps(source_revisions,indent=2)+'\n')
files=[]
for path in sorted(p.STORE.rglob('*')):
    parts=path.relative_to(p.STORE).parts
    if any(x.startswith('._') or x.endswith('.pending') for x in parts):continue
    if any(x.startswith('launch_') for x in parts):continue
    try:
        if path.is_file():files.append(p.fact(path))
    except FileNotFoundError:continue
manifest={'schema':'ddm_psa2_live_retention_snapshot.v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'root':str(p.STORE),'files':files,'stable_files_logical_bytes':sum(x['bytes'] for x in files),
 'storage_including_live_logs':p.capacity(),'artifact_references_verified':checks,
 'excluded':['._*','*.pending','launch_* live logs/control files'],'coverage':'point-in-time; later completed search blocks not included',
 'no_deletion':True,'charter_complete':False}
(out/'RETENTION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'files':len(files),'verified_refs':checks,'retained_stable_bytes':manifest['stable_files_logical_bytes'],'storage':p.capacity()}),flush=True)
PY
