"""Retain real stratified geometry vectors, tokens and raws on the charter SSD."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
from ddm_mrs1_validate import fact, atomic_json
from ddm_mrs2_profile import render_pair
from ddm_mrs4_proof import load_module, native_record, restore_pair, compare_bits
ROOT = Path('/Volumes/APDataStore/pact/ddm_mrs5')
BANK = Path('/Volumes/APDataStore/pact/ddm_mrs2/profile_python/RESULT.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', required=True, type=Path)
    parser.add_argument('--receiver', required=True, type=Path)
    args = parser.parse_args()
    run, receiver = args.resume_from.resolve(), args.receiver.resolve()
    if not run.is_relative_to(ROOT) or not receiver.is_relative_to(ROOT):
        raise ValueError('proof paths must stay in charter store')
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if shutil.disk_usage(ROOT).free < 1024**3:
        raise ValueError('1 GiB free required')
    sys.dont_write_bytecode = True
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260916)
    np.random.seed(20260916)
    bank = json.loads(BANK.read_text())
    rng = np.random.default_rng(20260916)
    sample = [int(i) for start in range(0,600,25) for i in sorted(rng.choice(25,2,replace=False)+start)]
    if [r['frame'] for r in bank['rows']] != sample:
        raise ValueError('sample differs')
    sources = [fact(p) for p in sorted(receiver.parent.iterdir()) if p.is_file() and not p.name.startswith('._')]
    binding = dict(sources=sources, bank=fact(BANK), runner=fact(Path(__file__)),
        helpers=[fact(Path(__file__).with_name(n)) for n in ('ddm_mrs1_validate.py','ddm_mrs2_profile.py','ddm_mrs4_proof.py')],
        axis='[macOS-CPU advisory]', seed=20260916, score_claim=False)
    inputs = run/'INPUTS.json'
    if inputs.exists() and json.loads(inputs.read_text()) != binding:
        raise ValueError('source binding changed')
    atomic_json(inputs,binding)
    module = load_module(receiver,'geometry_proof_receiver')
    if any(getattr(module,n) is None for n in ('_RANGE_LIBRARY','_CORRECTOR_LIBRARY','_GEOMETRY_LIBRARY')):
        raise ValueError('native set incomplete')
    record_type = native_record(receiver.with_name('corrector.c').read_text())
    vectors=[]
    class ShadowGeometry:
        def __init__(self, previous, config):
            self.native=module.NativeGeometry(previous,config)
            self.python=module.CausalGeometry(previous,config)
            self.plane=self.native.plane
        def contexts(self, positions):
            a,b=self.native.contexts(positions),self.python.contexts(positions)
            vectors.append((a.copy(),b.copy()))
            compare_bits(a,b,'geometry bins')
            return a
        def observe(self, positions, symbols):
            self.native.observe(positions,symbols)
            self.python.observe(positions,symbols)
            compare_bits(self.native.plane,self.python.plane,'geometry plane')
    module.geometry_mixer_LaneGeometry=ShadowGeometry
    rows=[]
    with torch.inference_mode():
        parts,model,basis,coefficients,selector=module.read_models(receiver.with_name('archive.zip'))
        basis=module.render_normalized_basis(basis)
        modes,labels=module.selector_decode_selector(selector)
        for original in bank['rows']:
            index=original['frame']; receipt=run/f'pair_{index:04d}.json'
            if fact(Path(original['state']['path'])) != original['state']:
                raise ValueError('state changed')
            if receipt.exists():
                row=json.loads(receipt.read_text())
                if row['binding']!=binding or row['frame']!=index or row['state']!=original['state']:
                    raise ValueError('resume binding differs')
                for item in row['artifacts']:
                    if fact(Path(item['path']))!=item: raise ValueError('retained artifact changed')
                if row['token']['sha256']!=original['token']['sha256'] or row['raw']['sha256']!=original['raw']['sha256']:
                    raise ValueError('resumed oracle differs')
                rows.append(row); continue
            decoder=module.TokenDecoder(parts,torch.device('cpu'))
            restore_pair(Path(original['state']['path']),decoder,module,record_type)
            vectors.clear()
            try:
                token=next(decoder)
            finally:
                path=run/f'pair_{index:04d}.geometry.npz'
                if vectors: np.savez_compressed(path,native=np.concatenate([v[0] for v in vectors]),python=np.concatenate([v[1] for v in vectors]))
            token_path=run/f'pair_{index:04d}.tokens.u8'; token_path.write_bytes(token.numpy().tobytes())
            pair,_,_=render_pair(module,model,basis,coefficients,modes,labels,token,index)
            raw=run/f'pair_{index:04d}.raw'; raw.write_bytes(pair.tobytes())
            tf,rf=fact(token_path),fact(raw)
            if tf['sha256']!=original['token']['sha256'] or rf['sha256']!=original['raw']['sha256']:
                raise ValueError('sample token/raw differs; bytes retained')
            row=dict(binding=binding,frame=index,state=original['state'],token=tf,raw=rf,artifacts=[tf,rf,fact(path)],groups=len(vectors),matched=True)
            if row['groups']!=190: raise ValueError('incomplete geometry comparison')
            atomic_json(receipt,row); rows.append(row)
            decoder.corrector.close()
            print(json.dumps(dict(completed_pairs=len(rows),frame=index)),flush=True)
    atomic_json(run/'RESULT.json',dict(binding=binding,rows=rows,sample=sample,matched_pairs=len(rows),geometry_groups=sum(r['groups'] for r in rows)))


if __name__=='__main__': main()
