"""Retain coverage of cold-start branches in addition to the seeded n600 sample."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from ddm_mrs1_validate import ARCHIVE, ROOT, atomic_json, fact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receiver', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path, required=True)
    args = parser.parse_args()
    receiver, run = args.receiver.resolve(), args.resume_from.resolve()
    if not receiver.is_relative_to(ROOT) or not run.is_relative_to(ROOT):
        raise ValueError('trace must remain in the charter store')
    run.mkdir(parents=True, exist_ok=True)
    binding = dict(receiver=fact(receiver), archive=fact(ARCHIVE), seed=20260916,
                   runner=fact(Path(__file__).resolve()), axis='[macOS-CPU advisory]')
    reference_path = ROOT / 'python_proof4/tokens.u8'
    reference_inputs = ROOT / 'python_proof4/INPUTS.json'
    if json.loads(reference_inputs.read_text())['archive'] != binding['archive']:
        raise ValueError('reference belongs to another archive')
    with reference_path.open('rb') as stream:
        first_pair = stream.read(384 * 512)
    binding['reference'] = dict(path=str(reference_path), first_pair_bytes=len(first_pair),
        first_pair_sha256=hashlib.sha256(first_pair).hexdigest(), inputs=fact(reference_inputs))
    if len(first_pair) != 384 * 512:
        raise ValueError('reference first pair is incomplete')
    done = run / 'COMPLETE.json'
    if done.exists():
        receipt = json.loads(done.read_text())
        if receipt['binding'] != binding or fact(Path(receipt['token']['path'])) != receipt['token']:
            raise ValueError('completed cold-start trace binding changed')
        return
    random.seed(20260916)
    np.random.seed(20260916)
    torch.manual_seed(20260916)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    sys.dont_write_bytecode = True
    lines = set()

    def trace(frame, event, argument):
        if frame.f_code.co_filename != str(receiver):
            return None
        if event == 'line':
            lines.add(frame.f_lineno)
        return trace

    sys.settrace(trace)
    try:
        spec = importlib.util.spec_from_file_location('receiver', receiver)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        with torch.inference_mode():
            parts, *_ = module.read_models(ARCHIVE)
            token = next(module.TokenDecoder(parts, torch.device('cpu'))).numpy()
    finally:
        sys.settrace(None)
    path = run / 'first_pair.npy'
    with path.with_suffix('.pending').open('wb') as stream:
        np.save(stream, token, allow_pickle=False)
    path.with_suffix('.pending').replace(path)
    reference = np.frombuffer(first_pair, dtype=np.uint8).reshape(384, 512)
    if not np.array_equal(token, reference):
        raise ValueError('cold-start token differs; retained for diagnosis')
    atomic_json(done, dict(binding=binding, token=fact(path), lines=sorted(lines), completed_pairs=[0],
                          note='Cold-start control, additional to the 32 seeded pairs', score_claim=False))


if __name__ == '__main__':
    main()
