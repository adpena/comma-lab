#!/usr/bin/env python3
"""Lossless RC3/TC1 composition on a pinned live archive; no scorer or dispatch.

Streaming encode mirrors ddm_jg2_tail_reencode's sparse public HPAC trajectory.
TC1 weights and mechanism are reused unchanged, not fitted on the new field.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from experiments import ddm_jg2_tail_reencode as jg2
from experiments import ddm_tc1_mixer_codec as tc1

ROOT = Path('/Volumes/VertigoDataTier/pact/ddm_cmp1_compose')
BULK = Path('/Volumes/APDataStore/pact/ddm_cmp1_compose')
LIVE = Path('/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/candidate_pass4/candidate_runtime')
RC3 = Path('/Volumes/VertigoDataTier/pact/ddm_rc3_shared_mixer_successor/candidate_runtime')
TC1 = Path('/Volumes/VertigoDataTier/pact/ddm_tc1_tail_shared_mixer/rebase_pc2')
FIELD = LIVE.parent.parent / 'admission_pass4/field_admitted.npz'
POINTER = REPO / '.omx/state/canonical_frontier_pointer.json'
AXIS = '[macOS-CPU advisory / scorer-free EXACT byte measurement]'


def preflight(path, need=1024 * 1024):
    """Check df before writing, enforce the charter's total Vertigo budget."""
    path = Path(path).resolve()
    root = ROOT if path.is_relative_to(ROOT) else BULK if path.is_relative_to(BULK) else None
    if root is None:
        raise ValueError(f'write outside owned stores: {path}')
    mount = Path('/Volumes') / path.parts[2]
    result = subprocess.run(['df', '-h', str(mount)], capture_output=True, text=True, check=True)
    if shutil.disk_usage(mount).free < need + 64 * 1024**2:
        raise RuntimeError(result.stdout + 'storage preflight refused')
    if root == ROOT:
        used = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) if ROOT.exists() else 0
        if used + need >= 1024**3:
            raise RuntimeError('Vertigo 1 GiB arm budget refused; route payload blobs to APDataStore')
    path.parent.mkdir(parents=True, exist_ok=True)


def write(path, payload):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'immutable payload changed: {path}')
        return jg2.file_fact(path)
    preflight(path, len(payload))
    jg2.atomic_write(path, payload)
    return jg2.file_fact(path)


def record(path, value):
    preflight(path)
    jg2.atomic_json(path, value)
    return value


def guard():
    inputs = json.loads((ROOT / 'INPUTS.json').read_text())
    live = json.loads(POINTER.read_text())['our_local_frontier_contest_cuda']
    if live['archive_sha256'] != inputs['archive']['sha256']:
        raise RuntimeError('POINTER_MOVED: refuse; rebase into a new generation')
    for name in ('archive', 'field', 'weights'):
        if jg2.file_fact(Path(inputs[name]['path'])) != inputs[name]:
            raise RuntimeError(f'{name} custody changed')
    return inputs


def initialize():
    if (ROOT / 'INPUTS.json').exists():
        return guard()
    pointer = json.loads(POINTER.read_text())
    base = jg2.file_fact(LIVE / 'archive.zip')
    if base['sha256'] != pointer['our_local_frontier_contest_cuda']['archive_sha256']:
        raise RuntimeError('configured source is not live')
    sources = {}
    for path in sorted(LIVE.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.so', '.dylib', '.pyc'):
            relative = path.relative_to(LIVE)
            sources[str(relative)] = write(ROOT / 'source_runtime' / relative, path.read_bytes())
    parts = jg2.split_member(jg2.read_archive_member(LIVE / 'archive.zip'))
    census = {k: write(ROOT / 'retained/source' / (k + '.bin'), v) for k, v in parts.items()}
    null = ROOT / 'retained/null_archive.zip'
    preflight(null)
    jg2.pack_archive(jg2.join_member(parts), null)
    if null.read_bytes() != (LIVE / 'archive.zip').read_bytes():
        raise RuntimeError('null rebuild differs from current archive')
    with np.load(FIELD, allow_pickle=False) as data:
        if set(data.files) != {str(i) for i in range(600)}:
            raise ValueError('admitted field does not contain all 600 planes')
        field = np.stack([data[str(i)] for i in range(600)])
    if field.shape != (600, 384, 512) or field.dtype != np.uint8 or field.max() >= 5:
        raise ValueError('field geometry/alphabet changed')
    field_fact = write(ROOT / 'retained/source/field.u8', field.tobytes())
    weights = write(ROOT / 'retained/weights_i8.bin', (TC1 / 'mixer/weights_i8.bin').read_bytes())
    rc = jg2.split_member(jg2.read_archive_member(RC3 / 'archive.zip'))
    old_tc = jg2.split_member(jg2.read_archive_member(TC1 / 'source_runtime/archive.zip'))
    new_tc = jg2.split_member(jg2.read_archive_member(TC1 / 'candidate_runtime/archive.zip'))
    if parts['hpac'] != old_tc['hpac']:
        raise RuntimeError('model base changed; old rc3 saving does not transfer')
    model_saving = len(parts['hpac']) - len(rc['hpac'])
    old_tail_saving = len(old_tc['tail']) - len(new_tc['tail'])
    if (model_saving, old_tail_saving) != (201, 549):
        raise RuntimeError('charter source savings failed verification')
    write(ROOT / 'retained/rc3_hpac.br', rc['hpac'])
    result = dict(schema='ddm_cmp1_inputs.v1', axis=AXIS, score_claim=False, seed=20260909,
                  archive=sources['archive.zip'], source_archive=base, field=field_fact,
                  original_field_npz=jg2.file_fact(FIELD), weights=weights, sources=sources,
                  pointer=pointer, census=census, null_archive=jg2.file_fact(null),
                  verified_model_saving=model_saving, predecessor_tail_saving=old_tail_saving,
                  expected_composed_bytes=base['bytes']-model_saving-old_tail_saving,
                  predicted_saving_band=[730,770], falsifier_saving_below=700,
                  storage_policy='Vertigo <1 GiB; APDataStore payload blobs only; retain all payloads')
    return record(ROOT / 'INPUTS.json', result)


def encode(tag, stop, resume_from):
    """Independent sparse traversal, mixed encode and original-stream control."""
    import torch
    inputs = guard()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(inputs['seed']); np.random.seed(inputs['seed']); random.seed(inputs['seed'])
    torch.use_deterministic_algorithms(True)
    work = ROOT / 'encode' / tag
    preflight(work / 'build')
    route = jg2.load_route_b()
    library, build = jg2.compile_rc64(work, route, 'cmp1_' + tag)
    residual, renderer, renderer_dir = jg2.load_runtime(ROOT / 'source_runtime')
    from runtime.free_corrector import FreeCorrector
    from runtime.hpac_inference import optimize_sparse_evaluator
    parts = residual.read_residual_archive(ROOT / 'source_runtime/archive.zip')
    target = jg2.load_tokens(Path(inputs['field']['path']))
    model = renderer.load_hpac(residual.materialize_ihs1(parts.hpac_blob, renderer), torch.device('cpu'))
    sparse = residual._sparse_class(renderer_dir)(model, 384, 512)
    corrector, cold = FreeCorrector(384*512), FreeCorrector(384*512)
    mixer = tc1.SharedMixer(Path(inputs['weights']['path']).read_bytes())
    binding = dict(inputs_sha=jg2.sha256_file(ROOT/'INPUTS.json'), source_sha=jg2.sha256_file(Path(__file__)),
                   codec_sha=jg2.sha256_file(Path(tc1.__file__)), jg2_sha=jg2.sha256_file(Path(jg2.__file__)),
                   runtime_sources={k:v['sha256'] for k,v in inputs['sources'].items()}, build=build)
    for relative, fact in inputs['sources'].items():
        if jg2.file_fact(ROOT/'source_runtime'/relative) != fact:
            raise RuntimeError('source runtime drift')
    start = 0
    per_frame = np.zeros((2,600), dtype=np.float64)
    latest = work / 'LATEST.json'
    state = None
    if latest.exists():
        receipt = json.loads(latest.read_text())
        if receipt['binding'] != binding or jg2.file_fact(Path(receipt['payload']['path'])) != receipt['payload']:
            raise RuntimeError('checkpoint binding drift')
        with np.load(receipt['payload']['path'],allow_pickle=False) as data:
            state = {k:data[k] for k in data.files}
        start = int(state['frame'][0])
        jg2.load_corrector_state(corrector,{k[2:]:v for k,v in state.items() if k.startswith('c_')})
        mixer.restore({k[2:]:v for k,v in state.items() if k.startswith('m_')})
        if mixer.frame != start:
            raise RuntimeError('mixer checkpoint boundary mismatch')
        per_frame = state['per_frame']
    if not 0 <= start <= stop <= 600:
        raise ValueError('invalid frame range')
    encoders = [route.NativeRc64Encoder(library, None if state is None else state[k].tobytes())
                for k in ('control_encoder','mixed_encoder')]
    previous = torch.zeros((1,384,512),dtype=torch.long)
    if start:
        previous[0] = torch.from_numpy(target[start-1].astype(np.int64))
    groups = [np.flatnonzero(m.cpu().numpy().reshape(-1)) for m in renderer.group_masks(torch.device('cpu'))]
    started = time.monotonic()
    with torch.inference_mode():
        optimize_sparse_evaluator(sparse)
        for frame in range(start,stop):
            previous_cpu = None if frame == 0 else previous[0].numpy().astype(np.uint8)
            boundary = np.full(384*512,4,dtype=np.uint8) if frame == 0 else residual._boundary_buckets(previous_cpu).reshape(-1)
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(torch.tensor([frame]),previous)
            corrector.begin_frame(boundary); mixer.begin_frame()
            truth = np.asarray(target[frame]).reshape(-1)
            bits = [0.,0.]
            for group, positions in enumerate(groups):
                logits = sparse.selected_logits(current,context,group).cpu().numpy()
                predicted = logits.argmax(axis=1).astype(np.int64)
                feature = boundary[positions].astype(np.int64)*5+predicted
                probability = residual._probability_table(logits+parts.table.values[feature],renderer.HPAC_LOGIT_PRECISION)
                cs = corrector.group_state(probability,predicted,positions)
                original = corrector.coding_row(cs)
                coding = mixer.coding(original,positions,current[0].numpy().astype(np.uint8),previous_cpu)
                symbols = truth[positions].astype(np.int32)
                for i, rows in enumerate((original,coding)):
                    encoders[i].encode(symbols,rows)
                    freq = tc1.frequencies(rows)
                    bits[i] += float(-np.log2(freq[np.arange(len(symbols)),symbols].astype(float)/tc1.TOTAL).sum())
                corrector.observe(cs,symbols.astype(np.int64))
                current.reshape(-1)[torch.from_numpy(positions)] = torch.from_numpy(symbols.astype(np.int64))
            plane = current[0].numpy().astype(np.uint8)
            np.testing.assert_array_equal(plane,target[frame])
            corrector.end_frame(plane.reshape(-1)); mixer.end_frame(plane,previous_cpu)
            previous = current
            per_frame[:,frame] = bits
            if (frame+1)%10 == 0 or frame+1 == stop:
                captured = jg2.corrector_state(corrector)
                if jg2.uncaptured_divergent_state(corrector,cold,set(captured)):
                    raise RuntimeError('checkpoint would omit mutable corrector state')
                values = {'c_'+k:v for k,v in captured.items()}
                values.update({'m_'+k:v for k,v in mixer.snapshot().items()})
                values.update(frame=np.array([frame+1]),per_frame=per_frame.copy(),
                              control_encoder=np.frombuffer(encoders[0].snapshot(),dtype=np.uint8),
                              mixed_encoder=np.frombuffer(encoders[1].snapshot(),dtype=np.uint8))
                payload = BULK / 'encode' / tag / f'stage_{frame+1:04d}.npz'
                preflight(payload, sum(v.nbytes for v in values.values()))
                if payload.exists():
                    with np.load(payload,allow_pickle=False) as old:
                        if set(old.files)!=set(values) or any(not np.array_equal(old[k],v) for k,v in values.items()):
                            raise RuntimeError('immutable checkpoint differs')
                else:
                    temporary = payload.with_suffix('.npz.new')
                    with temporary.open('wb') as f:
                        np.savez_compressed(f,**values); f.flush(); os.fsync(f.fileno())
                    os.replace(temporary,payload)
                receipt = dict(binding=binding,frame=frame+1,payload=jg2.file_fact(payload))
                record(work/f'STAGE_{frame+1:04d}.json',receipt); record(latest,receipt)
                print(json.dumps(dict(tag=tag,frame=frame+1,seconds=time.monotonic()-started)),flush=True)
    outputs = []
    for name, encoder in zip(('control','mixed'),encoders):
        write(work/f'{name}_{stop:04d}.envelope',encoder.finish())
        raw = ctypes.string_at(encoder.library.rc64_encoder_data(encoder.context),int(encoder.library.rc64_encoder_size(encoder.context)))
        outputs.append(write(work/f'{name}_{stop:04d}.rc64',raw))
    identical = stop == 600 and Path(outputs[0]['path']).read_bytes() == parts.token_stream
    result = dict(axis=AXIS,score_claim=False,frames=stop,start_frame=start,tag=tag,
                  control=outputs[0],mixed=outputs[1],control_identical=identical,
                  ideal_bytes=(per_frame.sum(axis=1)/8).tolist(),binding=binding)
    record(work/f'ENCODE_{stop:04d}.json',result)
    if stop == 600 and not identical:
        raise RuntimeError('full current-pointer original-coder control failed')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['init','encode'])
    parser.add_argument('--tag', choices=['primary','repeat','resume_control'],default='primary')
    parser.add_argument('--stop',type=int,default=600)
    parser.add_argument('--resume-from',type=Path,required=True)
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT:
        raise ValueError('resume-from must name the bound arm store')
    print(json.dumps(initialize() if args.stage=='init' else encode(args.tag,args.stop,args.resume_from),sort_keys=True))
