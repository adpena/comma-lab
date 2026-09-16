#!/bin/sh
set -eu
cd /Users/adpena/Projects/pact
exec .venv/bin/python - <<'PY'
import datetime,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,'experiments')
import ddm_psa1_byte_price as p
root=p.STORE
out=Path('.omx/research/ddm_psa1_20260916')
print(json.dumps({'storage_before_heavy':p.capacity(4<<20)}),flush=True)
forms=['A','B3','C','D','B5_control']
summary=[]
verified=set()
checks=0
sample=p.load(root/'SAMPLE.json')
exchange=25/37545489
for form in forms:
    done=Path('.omx/tmp/codex_runs')/('ddm_psa1_'+form+'.done')
    d=p.load(done)
    assert d['rc']==0,(form,d)
    (out/('ddm_psa1_'+form+'.done.json')).write_text(json.dumps(d,indent=2)+'\n')
    records=p.load(root/'prices'/form/'RESULTS.json')
    assert len(records)==38
    pairs=[r for r in records if r['pair'] is not None]
    assert [r['pair'] for r in pairs]==sample['pairs'] and len(set(sample['pairs']))==36
    for r in records:
        assert r['twins_identical'] and r['exact_code_parseback'] and not r['receiver_implemented']
        for f in r['artifacts']:
            assert p.fact(Path(f['path']))==f,f
            verified.add(f['path']); checks+=1
    null=next(r for r in records if r['label']=='empty')
    joint=next(r for r in records if r['label']=='joint36')
    legs={}
    for key in ['standalone_q11_bytes','sm1_append_bytes','sm1_archive_delta_bytes']:
        values=np.array([r[key] for r in pairs])
        payable=[r['pair'] for r in pairs if r[key]<r['debt']['debt_bytes']]
        calibrated=[r['pair'] for r in pairs if r[key]<r['debt']['cells']*(100*0.00010304/12147)/exchange]
        legs[key]={'median':float(np.median(values)),'q1':float(np.quantile(values,.25)),
                   'q3':float(np.quantile(values,.75)),'min':int(values.min()),'max':int(values.max()),
                   'k':len(payable),'n':36,'payable_pairs':payable,'calibrated_T4_carry_k':len(calibrated),
                   'fixed_empty_bytes':null[key],'joint36_bytes':joint[key]}
    eligible=form!='B5_control'
    passed=eligible and any(legs[k]['k']>=4 and legs[k]['fixed_empty_bytes']<200 for k in ['standalone_q11_bytes','sm1_archive_delta_bytes'])
    summary.append({'form':form,'eligible':eligible,'passes_complete_singleton_price_gate':passed,
                    'legs':legs,'header_bytes':12,'index_bytes_per_pair':2,'standalone_footer_bytes':8,
                    'joint36_raw_header_index_footer_bytes':12+72+8,'research_only':True})
allfiles=[]
for file in sorted(root.rglob('*')):
    if any(x.startswith('._') or x.endswith('.pending') for x in file.parts):continue
    if 'launch_harvest' in file.relative_to(root).parts:continue
    try:
        if file.is_file():allfiles.append(p.fact(file))
    except FileNotFoundError:continue
retained=sum(f['bytes'] for f in allfiles)
assert retained<=1<<30
manifest={'schema':'ddm_psa1_retention.v1','root':str(root),'files':allfiles,'logical_bytes':retained,
          'files_n':len(allfiles),'skip':['._*','*.pending','live launch_harvest control logs; final receipt saved separately'],'cleanup':'Nothing deleted; immutable resume reuse and certify-or-block quota.'}
(out/'RETENTION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
inputs=p.load(root/'INPUTS.json')
source_checks=[]
for item in inputs['runtime']:
    assert p.fact(Path(item['source']['path']))==item['source']
    assert p.fact(Path(item['copy']['path']))==item['copy']
    source_checks.append(item)
for name,expected in inputs['sources'].items():
    assert p.fact(Path(expected['path']))==expected,(name,expected)
assert p.fact(Path(inputs['argmax']['path']))==inputs['argmax']
assert p.fact(Path(inputs['gt']['path']))==inputs['gt']
(out/'SOURCE_INTEGRITY.json').write_text(json.dumps({'runtime_files_verified':len(source_checks),'copied_and_original_identical':True,'argmax_and_gt_hashes_reverified':True,'source_receipts_reverified':True,'input_receipt':p.fact(root/'INPUTS.json')},indent=2)+'\n')
result={'schema':'ddm_psa1_price_gate.v1','axis':p.AXIS,'score_claim':False,'research_only':True,
        'archive_sha256':p.PIN,'archive_bytes':179332,'pairs':sample['pairs'],'selection':sample['selection'],
        'population_residual_cells':sum(sample['population_counts']),'population_median_debt_bytes':sample['median_population_debt_bytes'],
        'S_per_byte':exchange,'S_per_cell_advisory':100/(600*384*512),'forms':summary,
        'gate_passed':any(r['passes_complete_singleton_price_gate'] for r in summary),
        'verdict_scope':'STEP0 FORMULATION price screen, four forms with 36 minimum-nonzero commands per form on move52; no efficacy or family closure',
        'archive_objects':38*5*2*2,'artifact_checks':checks,'distinct_artifact_paths_verified':len(verified),
        'retention':{'bytes':retained,'files':len(allfiles),'limit_bytes':1<<30},
        'tail_context':'NOT_APPLICABLE: shipped K=5 HPAC geometry does not accept int4 vectors',
        'boundaries':['no scorer','no renderer forward','no pose solve','no training','no Modal','no fire','no packet','no authorize_*','no receiver edit','no upstream edit','no PR/sealed tree edit','no predecessor store writes','no payload deletion','no staged-index edits','no score or frontier movement'],
        'unmeasured':['effective repair values','seg benefit','pose effects','all600 collateral','public receiver implementation','contest score'],
        'producer':p.fact(Path(p.__file__)),'inputs':p.fact(root/'INPUTS.json'),'sample':p.fact(root/'SAMPLE.json'),
        'retention_manifest':p.fact(out/'RETENTION_MANIFEST.json')}
(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'gate':result['gate_passed'],'forms':summary,'retained_bytes':retained}),flush=True)
PY
