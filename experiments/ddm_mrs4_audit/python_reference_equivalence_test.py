"""Independently rehash and compare the retained real-pair differential vectors."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def verify_fact(item):
    path = Path(item['path'])
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != item['sha256'] or path.stat().st_size != item['bytes']:
        raise ValueError('retained artifact changed: ' + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.resume_from / 'RESULT.json').read_text())
    binding = result['binding']
    if binding != json.loads((args.resume_from / 'INPUTS.json').read_text()) or binding['mode'] != 'parity':
        raise ValueError('the completed differential proof is required')
    verify_fact(binding['bank'])
    for item in binding['source'] + binding['helpers']:
        verify_fact(item)
    bank = json.loads(Path(binding['bank']['path']).read_text())
    rng = np.random.default_rng(20260916)
    expected = [int(i) for start in range(0, 600, 25) for i in sorted(rng.choice(25, 2, replace=False) + start)]
    if result['matched_pairs'] != 48 or [r['frame'] for r in result['rows']] != expected:
        raise ValueError('not the complete pre-registered stratified population')
    original = {r['frame']: r for r in bank['rows']}
    for row in result['rows']:
        reference = original[row['frame']]
        if row['state'] != reference['state'] or row['corrector_groups'] != 190 or row['corrector_state_arrays'] != 79:
            raise ValueError('state or comparison denominator differs')
        verify_fact(row['state'])
        for item in row['artifacts']:
            verify_fact(item)
        for key in ('token', 'raw'):
            if row[key]['sha256'] != reference[key]['sha256'] or row[key]['bytes'] != reference[key]['bytes']:
                raise ValueError('oracle identity differs')
        path = next(Path(item['path']) for item in row['artifacts'] if item['path'].endswith('.npz'))
        with np.load(path) as vectors:
            if set(vectors.files) != {'corrector_native', 'corrector_python', 'mixer_native', 'mixer_python'}:
                raise ValueError('differential vector set differs')
            for label, count in (('corrector', 384 * 512), ('mixer', 2 * 384 * 512)):
                native, python = vectors[label + '_native'], vectors[label + '_python']
                if native.dtype != np.float32 or python.dtype != np.float32 or native.shape != (count, 5) or python.shape != native.shape:
                    raise ValueError('differential vector shape or dtype differs')
                if native.tobytes() != python.tobytes():
                    raise ValueError('retained probability vectors differ')
    print(json.dumps(dict(matched_pairs=48, corrector_groups=9120, probability_rows=48 * 384 * 512 * 3,
                         axis='[macOS-CPU advisory]', score_claim=False)))


if __name__ == '__main__':
    main()
