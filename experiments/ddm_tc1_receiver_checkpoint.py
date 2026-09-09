"""Optional, source-bound frame checkpoints for the actual TC1 public reader.

No checkpoint input is used in normal inflation. Native structures below mirror
the frozen source ABI and expose persistent state only; scratch is rebuilt by
the existing begin_frame/group_state calls. Checkpoints never alter probabilities.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np


class Family(C.Structure):
    _fields_ = [(n, C.POINTER(C.c_int64)) for n in ('counts', 'hits', 'phat_q')] + [
        ('size', C.c_int64), ('count_limit', C.c_int64), ('rule', C.c_int)]


class CorrectorPrefix(C.Structure):
    _fields_ = [('plane', C.c_int64), ('prev1', C.POINTER(C.c_uint8)),
                ('prev2', C.POINTER(C.c_uint8)), ('run', C.POINTER(C.c_int64)),
                ('boundary', C.POINTER(C.c_int64)), ('have_prev', C.c_int),
                ('current', C.POINTER(C.c_uint8)), ('known', C.POINTER(C.c_uint8)),
                ('families', Family * 23), ('weights', C.POINTER(C.c_int64)),
                ('miss_counts', C.POINTER(C.c_int64)), ('miss_expect', C.POINTER(C.c_int64)),
                ('miss_seen', C.POINTER(C.c_int64)), ('capacity', C.c_int64),
                ('n', C.c_int64), ('group_open', C.c_int)]


class DecoderState(C.Structure):
    _fields_ = [('low', C.c_uint64), ('high', C.c_uint64), ('code', C.c_uint64),
                ('data', C.POINTER(C.c_uint8)), ('size', C.c_size_t),
                ('bit_position', C.c_size_t), ('error', C.c_int)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    os.replace(temporary, path)


class ReceiverCheckpoint:
    """Resume the same decoder state; source/build/stream drift refuses."""

    def __init__(self, parts, decoder, corrector, mixer, code_dir):
        self.root = Path(os.environ['TC1_RECEIVER_CHECKPOINT_DIR'])
        self.root.mkdir(parents=True, exist_ok=True)
        if type(corrector).__name__ != 'NativeFreeCorrector':
            raise ValueError('TC1 frame checkpoints require the frozen native corrector')
        corrector_source = Path(__file__).parent / 'f26_corrector_native.c'
        source = corrector_source.read_text()
        if '#define N_FAMILIES 23 ' not in source or corrector.library.f26_corrector_abi_version() != 1:
            raise ValueError('unknown corrector state ABI')
        self.corrector, self.decoder, self.mixer = corrector, decoder, mixer
        self.c = C.cast(corrector.handle, C.POINTER(CorrectorPrefix)).contents
        self.d = C.cast(decoder.context, C.POINTER(DecoderState)).contents
        if self.c.plane != 384 * 512 or self.d.size != len(parts.token_stream):
            raise ValueError('native state geometry differs')
        roots = [Path(code_dir), Path(__file__).parent]
        sources = sorted(set(p for r in roots for p in r.rglob('*') if p.is_file() and p.suffix in ('.py', '.c', '.sh')))
        self.binding = dict(schema='tc1_public_receiver_checkpoint.v1',
                            stream=hashlib.sha256(parts.token_stream).hexdigest(),
                            weights=hashlib.sha256(parts.tc1_weights).hexdigest(),
                            hpac=hashlib.sha256(parts.hpac_blob).hexdigest(),
                            residual=hashlib.sha256(parts.table.values.tobytes()).hexdigest(),
                            platform=platform.platform(),
                            sources={str(p): digest(p) for p in sources},
                            libraries={key: digest(os.environ[key]) for key in ('CPR1_RC64_LIBRARY', 'F26_CORRECTOR_NATIVE_LIBRARY')})
        self.loaded = None
        latest = self.root / 'LATEST.json'
        if latest.exists():
            receipt = json.loads(latest.read_text())
            if receipt['binding'] != self.binding or digest(receipt['path']) != receipt['sha256']:
                raise ValueError('receiver checkpoint binding or bytes changed')
            with np.load(receipt['path'], allow_pickle=False) as data:
                self.loaded = {k: data[k] for k in data.files}

    def arrays(self):
        c = self.c
        arrays = {name: np.ctypeslib.as_array(getattr(c, name), shape=(c.plane,))
                  for name in ('prev1', 'prev2', 'run', 'boundary', 'current', 'known')}
        for i, family in enumerate(c.families):
            if not 0 < family.size < 10000000:
                raise ValueError('invalid native family state size')
            for which, name in enumerate(('counts', 'hits', 'phat_q')):
                view = np.ctypeslib.as_array(getattr(family, name), shape=(family.size,))
                np.testing.assert_array_equal(view, self.corrector.table(which, i))
                arrays[f'family_{i}_{name}'] = view
        for which, name in ((3, 'weights'), (4, 'miss_counts'), (5, 'miss_expect'), (6, 'miss_seen')):
            expected = self.corrector.table(which)
            view = np.ctypeslib.as_array(getattr(c, name), shape=expected.shape)
            np.testing.assert_array_equal(view, expected)
            arrays[name] = view
        return arrays

    def restore(self, tokens, previous):
        if self.loaded is None:
            return 0
        state = self.loaded
        frame = int(state['frame'][0])
        if not 0 < frame <= 600 or state['tokens'].shape != (frame, 384, 512):
            raise ValueError('invalid receiver checkpoint frame geometry')
        arrays = self.arrays()
        expected_keys = {'corrector_' + name for name in arrays}
        actual_keys = {name for name in state if name.startswith('corrector_')}
        if actual_keys != expected_keys:
            raise ValueError('receiver checkpoint lost a corrector table')
        for name, view in arrays.items():
            value = state['corrector_' + name]
            if value.dtype != view.dtype or value.shape != view.shape:
                raise ValueError('receiver checkpoint table shape/dtype changed')
            view[:] = value
        self.c.have_prev = int(state['have_prev'][0])
        for name in ('low', 'high', 'code', 'bit_position', 'error'):
            setattr(self.d, name, int(state['decoder_' + name][0]))
        self.mixer.restore({k[len('mixer_'):]: v for k, v in state.items() if k.startswith('mixer_')})
        if self.mixer.frame != frame:
            raise ValueError('mixer and native checkpoint stages differ')
        import torch
        tokens[:frame] = torch.from_numpy(state['tokens'])
        previous[0] = torch.from_numpy(state['tokens'][-1].astype(np.int64)).to(previous.device)
        for name, view in self.arrays().items():
            np.testing.assert_array_equal(view, state['corrector_' + name])
        self.loaded = None
        return frame

    def save(self, frame, tokens):
        if self.c.group_open or self.d.error or self.mixer.frame != frame:
            raise ValueError('checkpoint is not at a completed healthy frame boundary')
        values = {'corrector_' + k: v.copy() for k, v in self.arrays().items()}
        values.update({'mixer_' + k: v for k, v in self.mixer.snapshot().items()})
        values.update({'decoder_' + k: np.array([getattr(self.d, k)], dtype=np.uint64)
                       for k in ('low', 'high', 'code', 'bit_position', 'error')})
        values.update(frame=np.array([frame]), have_prev=np.array([self.c.have_prev]),
                      tokens=tokens[:frame].numpy().copy())
        path = self.root / f'stage_{frame:04d}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                if set(old.files) != set(values) or any(not np.array_equal(old[k], v) for k, v in values.items()):
                    raise ValueError('receiver changed an immutable completed checkpoint')
        else:
            temporary = path.with_suffix('.npz.new')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **values)
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        receipt = dict(binding=self.binding, frame=frame, path=str(path), bytes=path.stat().st_size, sha256=digest(path))
        atomic_json(path.with_suffix('.json'), receipt)
        atomic_json(self.root / 'LATEST.json', receipt)
        print(json.dumps(dict(stage='public_token_decode', pairs=frame, checkpoint=receipt['sha256'])), flush=True)
        stop = int(os.environ.get('TC1_RECEIVER_STOP_AFTER', '600'))
        if frame >= stop and frame < 600:
            raise RuntimeError('TC1_RECEIVER_STAGE_COMPLETE')
