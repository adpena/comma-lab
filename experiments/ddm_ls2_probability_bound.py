#!/usr/bin/env python3
"""LS2 finite additive probabilities on retained real receiver rows; scorer-free.

Research instrument only. Produces no candidate and makes no score claim.
"""
from __future__ import annotations
import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from experiments.ddm_ls1_shipped_surprise import FIELD, FIELD_SHA, ARCHIVE_SHA, fact
from experiments.ddm_ls1_oracle_atlas import lane_row_distance, previous_frame_distance
from experiments.ddm_tc1_mixer_codec import frequencies, mix_probabilities, log2_fixed

ROOT = Path('/Volumes/VertigoDataTier/pact/ddm_ls2_full_resolution_lane_probability_bound')
LS1 = ROOT.parent / 'ddm_ls1'
N, H, W, K, TOTAL = 600, 384, 512, 5, 1 << 31
SIZE = H * W
Y, X = np.indices((H, W))
PHASE = (X % 64 + 2 * (Y % 64)).reshape(-1)
LEVELS = (8, 8, 190, 64)
FAMILIES = {'separate': (0, 1, 2), 'joint': (3, 2)}
LOW, HIGH = -4., 127 / 32
AXIS = '[macOS-CPU advisory / scorer-free n600 receiver probabilities]'


def guard(need=0):
    if shutil.disk_usage(ROOT).free < (16 << 30) + need:
        raise RuntimeError('STORAGE_BLOCK: all existing bytes retained')


def save(path, value):
    guard()
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError('write outside owned SSD')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.new')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    marker = path.with_suffix(path.suffix + '.sha256')
    temporary = marker.with_suffix(marker.suffix + '.new')
    temporary.write_text(digest + '\n')
    temporary.replace(marker)


def load_json(path):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != path.with_suffix(path.suffix + '.sha256').read_text().strip():
        raise ValueError('JSON checkpoint checksum mismatch: ' + str(path))
    return json.loads(data)


def payload(path, data):
    guard(len(data))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('immutable payload replay mismatch')
    else:
        tmp = path.with_suffix(path.suffix + '.new')
        tmp.write_bytes(data)
        tmp.replace(path)
    save(path.with_suffix(path.suffix + '.json'), fact(path))


def arrays(path, values):
    guard(sum(v.nbytes for v in values.values()))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(values) or any(not np.array_equal(old[k], v) for k, v in values.items()):
                raise ValueError('array replay mismatch: ' + str(path))
    else:
        tmp = path.with_suffix('.npz.new')
        with tmp.open('wb') as handle:
            np.savez_compressed(handle, **values)
        tmp.replace(path)
    save(path.with_suffix('.json'), fact(path))


def checked(path):
    if fact(path) != json.loads(path.with_suffix('.json').read_text()):
        raise ValueError('hash mismatch ' + str(path))
    return np.load(path, allow_pickle=False)


def bind():
    pointer = json.loads((REPO / '.omx/state/canonical_frontier_pointer.json').read_text())
    if pointer['effective_frontier']['archive_sha256'] != ARCHIVE_SHA:
        raise ValueError('POINTER_CHANGED: explicit rebind required')
    inputs = json.loads((LS1 / 'INPUTS.json').read_text())
    if fact(FIELD)['sha256'] != FIELD_SHA:
        raise ValueError('shipped field changed')
    archive = Path(inputs['sources']['archive.zip']['source']['path'])
    if fact(archive)['sha256'] != ARCHIVE_SHA:
        raise ValueError('archive drift')
    pins = {name: fact(LS1 / name) for name in ('TRACE.json', 'atlas/RESULT.json', 'verification/RESULT.json', 'INPUTS.json', 'MANIFEST.json')}
    pins['producer'] = fact(Path(__file__))
    pins['objective'] = fact(REPO / 'experiments/ddm_tc1_logistic_bound.c')
    pins['codec'] = fact(REPO / 'experiments/ddm_tc1_mixer_codec.py')
    pins['ls1_trace_source'] = fact(REPO / 'experiments/ddm_ls1_shipped_surprise.py')
    pins['ls1_atlas_source'] = fact(REPO / 'experiments/ddm_ls1_oracle_atlas.py')
    pins['protocol'] = fact(ROOT / 'PROTOCOL.md')
    path = ROOT / 'BINDING.json'
    if path.exists() and json.loads(path.read_text()) != pins:
        raise ValueError('source/input binding drift')
    save(path, pins)
    return pins


def read(frame):
    with checked(LS1 / 'rows' / f'frame_{frame:04d}.npz') as d:
        freq, truth, bits = d['frequencies'], d['symbols'], d['bits']
    with checked(LS1 / 'atlas/cells' / f'frame_{frame:04d}.npz') as d:
        keys = d['receiver_lane']
    past, visible = ((keys // 8) % 8).astype(np.int64), (keys % 8).astype(np.int64)
    return freq, truth, bits, (past, visible, PHASE, past * 8 + visible)


def tables(counts, expected):
    return [log2_fixed(np.clip((c + .5) / (e.astype(np.float64) / TOTAL + .5), 1/16, 16))
            for c, e in zip(counts, expected, strict=True)]


def features(freq, codes, table):
    winner = freq.argmax(axis=1)
    return np.stack([t[winner * levels + code] for t, levels, code in zip(table, LEVELS, codes, strict=True)], axis=2)


def prepare():
    bind()
    guard(4 << 30)
    rng = np.random.default_rng(20260911)
    held = np.sort(rng.choice(N, 120, replace=False))
    save(ROOT / 'SPLIT.json', dict(seed=20260911, heldout=held.tolist(), fit=[i for i in range(N) if i not in held]))
    start = 0
    counts = [np.zeros((K*l, K), dtype=np.int64) for l in LEVELS]
    expected = [np.zeros_like(c) for c in counts]
    states = sorted(p for p in (ROOT / 'prepare').glob('state_*.npz') if p.with_suffix('.json').exists())
    totals = np.zeros(4)
    if states:
        with checked(states[-1]) as d:
            start = int(d['frame'][0]); totals = d['totals']
            counts = [d[f'counts_{j}'] for j in range(4)]
            expected = [d[f'expected_{j}'] for j in range(4)]
    for frame in range(start, N):
        freq, truth, bits, codes = read(frame)
        tab = tables(counts, expected)
        phi = features(freq, codes, tab)
        arg = freq.argmax(axis=1).astype(np.uint8)
        keep = (TOTAL - freq.max(axis=1) >= TOTAL * 2**-14) | (truth != arg)
        arrays(ROOT / 'prepare' / f'frame_{frame:04d}.npz', dict(phi=phi[keep],
            logp=np.log(freq[keep].astype(np.float64) / TOTAL), truth=truth[keep], group=arg[keep],
            summary=np.array([bits.sum(),bits[keep].sum(),bits[~keep].sum(),keep.sum()]),
            **{f'table_{j}': t for j, t in enumerate(tab)}))
        totals += [bits.sum(), bits[keep].sum(), bits[~keep].sum(), keep.sum()]
        # No mutation until every feature in this complete plane has been materialized.
        for j, (levels, code) in enumerate(zip(LEVELS, codes, strict=True)):
            index = arg.astype(np.int64) * levels + code
            counts[j] += np.bincount(index*K+truth, minlength=levels*K*K).reshape(levels*K, K)
            for k in range(K):
                expected[j][:, k] += np.bincount(index, weights=freq[:, k], minlength=levels*K).astype(np.int64)
        # Complete disk restart at every plane, not just loop end.
        arrays(ROOT / 'prepare' / f'state_{frame+1:04d}.npz', dict(frame=np.array([frame+1]), totals=totals,
            **{f'counts_{j}': c for j,c in enumerate(counts)}, **{f'expected_{j}': e for j,e in enumerate(expected)}))
        if (frame+1) % 25 == 0:
            print(json.dumps(dict(stage='prepare', frames=frame+1, kept=int(totals[3]))), flush=True)
    save(ROOT / 'PREPARE.json', dict(pairs=N, symbols=N*SIZE, bits=totals[0], kept_bits=totals[1],
        omitted_bits=totals[2], kept_symbols=int(totals[3]), omitted_credit_bytes=totals[2]/8,
        axis=AXIS, counts_update='after each complete earlier plane'))


def objective(family, split):
    source = REPO / 'experiments/ddm_tc1_logistic_bound.c'
    libpath = ROOT / 'native/libtc1.dylib'
    if not (ROOT / 'native/BUILD.json').exists():
        libpath.parent.mkdir(exist_ok=True)
        cmd = ['cc', '-O3', '-std=c11', '-shared', '-fPIC', '-ffp-contract=off', '-fno-fast-math', str(source), '-o', str(libpath)]
        if libpath.exists():
            orphan = libpath.with_suffix('.unreceipted.dylib')
            if not orphan.exists(): shutil.copyfile(libpath, orphan)
        p = subprocess.run(cmd, capture_output=True, text=True, check=True)
        save(ROOT / 'native/BUILD.json', dict(argv=cmd, stdout=p.stdout, stderr=p.stderr, library=fact(libpath), source=fact(source)))
    elif json.loads((ROOT / 'native/BUILD.json').read_text())['library'] != fact(libpath):
        raise ValueError('native library drift')
    lib = ctypes.CDLL(str(libpath))
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    i16 = np.ctypeslib.ndpointer(dtype=np.int16, flags='C_CONTIGUOUS')
    u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
    lib.tc1_objective.argtypes = [ctypes.c_size_t,f64,i16,u8,u8,f64,ctypes.c_int,f64,f64]
    lib.tc1_objective.restype = ctypes.c_double
    ids = range(N) if split == 'full' else load_json(ROOT/'SPLIT.json')['fit']
    chunks = {k: [] for k in ('phi','logp','truth','group')}
    summary=np.zeros(4)
    for i in ids:
        with checked(ROOT/'prepare'/f'frame_{i:04d}.npz') as d:
            summary += d['summary']
            phi = np.zeros((len(d['truth']),K,7),dtype=np.int16)
            phi[:,:,:len(FAMILIES[family])] = d['phi'][:,:,FAMILIES[family]]
            chunks['phi'].append(phi)
            for k in ('logp','truth','group'): chunks[k].append(d[k])
    data = {k: np.ascontiguousarray(np.concatenate(v)) for k,v in chunks.items()}
    chunks.clear()
    def call(w, derivatives=True):
        grad=np.zeros((K,7)); hess=np.zeros((K,7,7))
        loss=lib.tc1_objective(len(data['truth']),data['logp'],data['phi'],data['truth'],data['group'],
            np.ascontiguousarray(w),int(derivatives),grad,hess)
        if not np.isfinite(loss) or not np.isfinite(grad).all(): raise ValueError('nonfinite objective')
        return loss,grad,hess
    # Independent NumPy categorical objective and derivatives on real population-strided rows.
    ix=np.linspace(0,len(data['truth'])-1,min(4096,len(data['truth'])),dtype=int)
    small={k:np.ascontiguousarray(v[ix]) for k,v in data.items()}
    w=np.full((K,7),.125); g=np.zeros((K,7)); h=np.zeros((K,7,7))
    loss=lib.tc1_objective(len(ix),small['logp'],small['phi'],small['truth'],small['group'],w,1,g,h)
    x=small['phi'].astype(float)*(math.log(2)/1024)
    z=small['logp']+np.einsum('nkf,nf->nk',x,w[small['group']])
    m=z.max(axis=1); e=np.exp(z-m[:,None]); prob=e/e.sum(axis=1)[:,None]
    expected_loss=float((np.log(e.sum(axis=1))+m-z[np.arange(len(ix)),small['truth']]).sum())
    gg=np.zeros_like(g); hh=np.zeros_like(h)
    for bank in range(K):
        mask=small['group']==bank; xx=x[mask]; pp=prob[mask]
        mean=np.einsum('nk,nkf->nf',pp,xx)
        gg[bank]=(mean-xx[np.arange(len(xx)),small['truth'][mask]]).sum(axis=0)
        centered=xx-mean[:,None,:]
        hh[bank]=np.einsum('nk,nkf,nkg->fg',pp,centered,centered)
    np.testing.assert_allclose(loss,expected_loss,rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose(g,gg,rtol=1e-8,atol=1e-8)
    np.testing.assert_allclose(h,hh,rtol=1e-8,atol=1e-8)
    save(ROOT/'fit'/family/split/'NUMERICAL_CONTROL.json',dict(real_rows=len(ix),status='PASS',
        loss_error=loss-expected_loss,gradient_max_error=float(np.max(abs(g-gg))),hessian_max_error=float(np.max(abs(h-hh)))))
    return call,summary


def fit(family, split):
    bind()
    folder=ROOT/'fit'/family/split
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'RESULT.json.sha256').exists():
        old = load_json(folder/'RESULT.json')
        if old['parameters'] != fact(folder/'parameters.bin'): raise ValueError('fit parameter drift')
        return
    obj,summary=objective(family,split)
    nf=len(FAMILIES[family]); w=np.zeros((K,7)); start=0
    iterations=sorted(p for p in folder.glob('ITER_*.json') if p.with_suffix('.json.sha256').exists())
    if iterations:
        d=load_json(iterations[-1]); w=np.array(d['weights']); start=d['iteration']
    if w.shape != (K,7) or not np.isfinite(w).all() or np.any(w[:,:nf]<LOW) or np.any(w[:,:nf]>HIGH):
        raise ValueError('invalid resumed weights')
    baseline=obj(np.zeros_like(w),False)[0]
    baseline_error=baseline/math.log(2)-summary[1]
    if abs(baseline_error)>.01: raise ValueError('zero objective reconciliation failed')
    for iteration in range(start,100):
        loss,grad,hess=obj(w)
        gap=float(np.sum(grad[:,:nf]*np.where(grad[:,:nf]>=0,w[:,:nf]-LOW,w[:,:nf]-HIGH)))
        receipt=dict(iteration=iteration,weights=w.tolist(),loss_nats=loss,baseline_nats=baseline,
            gap_bytes=gap/math.log(2)/8,lower_loss_nats=loss-gap,
            kept_gain_bytes=(baseline-loss)/math.log(2)/8,coefficient_bytes=K*nf,header_bytes=8,
            box=[LOW,HIGH],split=split,family=family,axis=AXIS,
            baseline_error_bits=float(baseline_error),omitted_credit_bytes=float(summary[2]/8),
            complete_baseline_bytes=float(summary[0]/8),
            optimistic_net_upper_bytes=float((summary[0]-(loss-gap)/math.log(2))/8-K*nf-8))
        save(folder/f'BOUND_{iteration:04d}.json',receipt)
        print(json.dumps(dict(stage='fit',family=family,split=split,iteration=iteration,
            gain_B=receipt['kept_gain_bytes'],gap_B=receipt['gap_bytes'])),flush=True)
        if gap/math.log(2)/8 < .05:
            rounded=np.rint(w[:,:nf]*32).clip(-128,127).astype(np.int8)
            data=b'LS2P'+bytes((1,int(family=='joint'),5,1))+rounded.tobytes()
            payload(folder/'parameters.bin',data)
            receipt['parameters']=fact(folder/'parameters.bin'); receipt['rounded']=rounded.tolist()
            save(folder/'RESULT.json',receipt)
            return
        direction=np.zeros_like(w)
        for k in range(K):
            a=hess[k,:nf,:nf]; g=grad[k,:nf]
            free=~(((w[k,:nf]<=LOW+1e-12)&(g>0))|((w[k,:nf]>=HIGH-1e-12)&(g<0)))
            idx=np.where(free)[0]
            if len(idx):
                sub=a[np.ix_(idx,idx)]
                direction[k,idx]=np.linalg.solve(sub+np.eye(len(idx))*max(1e-9,np.trace(sub)*1e-10),-g[idx])
        accepted=False
        for attempt,scale in enumerate(2.**-np.arange(20)):
            trial=w.copy(); trial[:,:nf]=np.clip(w[:,:nf]+scale*direction[:,:nf],LOW,HIGH)
            value=obj(trial,False)[0]
            save(folder/f'TRIAL_{iteration:04d}_{attempt:02d}.json',dict(weights=trial.tolist(),loss_nats=value,scale=float(scale)))
            if value <= loss+1e-4*float(np.sum(grad*(trial-w))):
                w=trial; accepted=True; break
        if not accepted: raise RuntimeError('line search stalled; retain current numerical bound')
        save(folder/f'ITER_{iteration+1:04d}.json',dict(iteration=iteration+1,weights=w.tolist()))
    raise RuntimeError('fit incomplete after 100 iterations')


def price():
    bind()
    models={}
    for family in FAMILIES:
        for split in ('train','full'):
            p=ROOT/'fit'/family/split/'parameters.bin'
            bound = load_json(p.parent/'RESULT.json')
            if fact(p) != bound['parameters']: raise ValueError('parameter receipt mismatch')
            d=p.read_bytes(); nf=len(FAMILIES[family])
            if d[:8] != b'LS2P'+bytes((1,int(family=='joint'),5,1)) or len(d)!=8+K*nf: raise ValueError('bad payload')
            models[family+'_'+split]=(family,np.frombuffer(d[8:],dtype=np.int8).reshape(K,nf))
    price_binding = dict(parameters={name: fact(ROOT/'fit'/family/split/'parameters.bin')
        for name, (family, split) in ((f+'_'+s, (f,s)) for f in FAMILIES for s in ('train','full'))},
        split=fact(ROOT/'SPLIT.json'), prepare=fact(ROOT/'PREPARE.json'), inputs=fact(ROOT/'BINDING.json'))
    bp=ROOT/'PRICE_BINDING.json'
    if bp.exists() and json.loads(bp.read_text()) != price_binding: raise ValueError('price resume binding drift')
    save(bp,price_binding)
    for frame in range(N):
        out=ROOT/'price'/f'frame_{frame:04d}.npz'
        if out.exists() and out.with_suffix('.json').exists():
            with checked(out): pass
            continue
        freq,truth,bits,codes=read(frame)
        with checked(ROOT/'prepare'/f'frame_{frame:04d}.npz') as d:
            tab=[d[f'table_{j}'] for j in range(4)]
        phi=features(freq,codes,tab)
        values={}
        for name,(family,w) in models.items():
            selected=phi[:,:,FAMILIES[family]]
            rows=mix_probabilities(freq,selected,w,original_rows=(freq.astype(np.float64)/TOTAL).astype(np.float32))
            ff=frequencies(rows)
            # Integer-row authority: zero added banks preserve original frequencies exactly.
            inactive=~np.any(w[freq.argmax(axis=1)],axis=1)
            ff[inactive]=freq[inactive]
            gain=(bits+np.log2(ff[np.arange(SIZE),truth]/TOTAL))/8
            values[name+'_class']=np.bincount(truth,weights=gain,minlength=K)
            values[name+'_row']=np.bincount(Y.reshape(-1),weights=gain,minlength=H)
            values[name+'_boundary_class']=np.bincount((Y.reshape(-1)%64==0).astype(int)*K+truth,weights=gain,minlength=2*K).reshape(2,K)
        arrays(out,values)
        if (frame+1)%25==0: print(json.dumps(dict(stage='price',frames=frame+1)),flush=True)
    held=load_json(ROOT/'SPLIT.json')['heldout']
    result={}
    for name in models:
        cs=[]; rs=[]; bs=[]
        for frame in range(N):
            with checked(ROOT/'price'/f'frame_{frame:04d}.npz') as d:
                cs.append(d[name+'_class']);rs.append(d[name+'_row']);bs.append(d[name+'_boundary_class'])
        cs=np.array(cs);rs=np.array(rs);bs=np.array(bs)
        arrays(ROOT/'tables'/f'{name}.npz',dict(pair_class=cs,pair_row=rs,pair_boundary_class=bs))
        charge=8+K*len(FAMILIES[models[name][0]])
        result[name]=dict(gross_bytes=float(cs.sum()),parameter_bytes=charge,net_bytes=float(cs.sum()-charge),
            class_bytes=cs.sum(axis=0).tolist(),patch_boundary_bytes=float(bs[:,1].sum()),other_rows_bytes=float(bs[:,0].sum()),
            heldout_pairs=120,heldout_gross_bytes=float(cs[held].sum()),heldout_class_bytes=cs[held].sum(axis=0).tolist(),
            fit_pairs=480,fit_gross_bytes=float(cs.sum()-cs[held].sum()))
    save(ROOT/'PRICING.json',dict(axis=AXIS,score_claim=False,full_n600=True,symbols=N*SIZE,models=result,
        meaning='integer RC64 log-loss; physical stream framing and archive build not measured'))


def controls():
    bind()
    target=np.memmap(FIELD,dtype=np.uint8,mode='r',shape=(N,H,W))
    ids=np.sort(np.random.default_rng(20260911).choice(N,32,replace=False))
    tested=0
    for frame in ids:
        done=ROOT/'controls'/f'frame_{frame:04d}.json'
        if done.with_suffix('.json.sha256').exists():
            tested += load_json(done)['mutation_positions']
            continue
        before=tested
        freq,truth,bits,codes=read(int(frame))
        np.testing.assert_array_equal(truth,target[frame].reshape(-1))
        np.testing.assert_array_equal(codes[0],previous_frame_distance(None if frame==0 else target[frame-1]))
        np.testing.assert_array_equal(codes[1],lane_row_distance(target[frame],True))
        for g in (0,1,31,63,64,126,189):
            changed=target[frame].copy(); changed.reshape(-1)[PHASE>=g]=5
            current=PHASE==g
            np.testing.assert_array_equal(lane_row_distance(changed,True)[current],codes[1][current])
            tested+=int(current.sum())
        zero=np.zeros((K,3),dtype=np.int8)
        ph=np.zeros((SIZE,K,3),dtype=np.int16)
        zz=frequencies(mix_probabilities(freq,ph,zero,original_rows=(freq.astype(float)/TOTAL).astype(np.float32)))
        zz[~np.any(zero[freq.argmax(axis=1)],axis=1)]=freq
        np.testing.assert_array_equal(zz,freq)
        selected=freq[np.arange(SIZE),truth]
        np.testing.assert_array_equal(bits,-np.log2(selected.astype(np.float64)/TOTAL))
        save(done,dict(frame=int(frame),mutation_positions=tested-before))
    save(ROOT/'CONTROLS.json',dict(status='PASS',seed=20260911,pairs=ids.tolist(),mutation_positions=tested,
        causality='future and same-group symbols masked; LS1 exact codes agree',counts='frozen until end of complete plane; state checkpoints retained'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','fit','price','controls'],required=True)
    parser.add_argument('--family',choices=list(FAMILIES))
    parser.add_argument('--split',choices=['train','full'])
    parser.add_argument('--resume-from',required=True)
    args=parser.parse_args()
    if Path(args.resume_from).resolve()!=ROOT: raise ValueError('wrong resume root')
    guard()
    # One owner per output store: also serializes shared binding/native creation.
    stage_lock=(ROOT/'.stage.lock').open('a+')
    fcntl.flock(stage_lock.fileno(),fcntl.LOCK_EX)
    started=time.monotonic()
    if args.stage=='fit': fit(args.family,args.split)
    else: globals()[args.stage]()
    save(ROOT/f'STAGE_{args.stage}_{args.family}_{args.split}.json',dict(argv=sys.argv,elapsed_s=time.monotonic()-started,axis=AXIS))
    retained=[fact(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ('.npz','.bin')]
    save(ROOT/f'RETENTION_{args.stage}_{args.family}_{args.split}.json',dict(files=retained,
        bytes=sum(p['bytes'] for p in retained),policy='KEEP all payloads and checkpoints; no deletion or move authorized',
        free_bytes=shutil.disk_usage(ROOT).free,reserve_bytes=16<<30))
    fcntl.flock(stage_lock.fileno(),fcntl.LOCK_UN)
    stage_lock.close()

if __name__=='__main__': main()
