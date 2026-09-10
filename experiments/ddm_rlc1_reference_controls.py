"""Resumable retained real-field geometry parity control; no score claim.

The initial completed control used the retained source copy under
reference_controls/run_controls.py. This landing adds explicit CLI routing.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path('/Users/adpena/Projects/pact')
FIELD = Path('/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/rebase_move40/move40/field.u8')
sys.path.insert(0, str(REPO))
from experiments.ddm_rlc1_geometry import LaneGeometry
from experiments.ddm_rlc1_reference import CONFIG_STRUCT, POSITIONS, ReferenceGeometry


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', type=Path, required=True)
    parser.add_argument('--concurrency-note', required=True)
    args = parser.parse_args()
    root = args.resume_from.resolve()
    root.mkdir(parents=True, exist_ok=True)
    facts = {str(p): sha(p) for p in (REPO / 'experiments/ddm_rlc1_geometry.c', REPO / 'experiments/ddm_rlc1_geometry.py', REPO / 'experiments/ddm_rlc1_reference.py', Path(__file__))}
    manifest = root / 'manifest.json'
    frames = sorted(np.random.default_rng(20260910).choice(600, 32, replace=False).tolist())
    config = CONFIG_STRUCT.pack(1, 128, 320, 16, 4, 4, 36, 2, 2, 3, 24, 0, 1, 2, 4, 8, 16)
    bound = {'sources': facts, 'frames': frames, 'seed': 20260910, 'field_sha256': sha(FIELD), 'config_hex': config.hex()}
    if manifest.exists():
        if json.loads(manifest.read_text()) != bound:
            raise ValueError('resume source/input/config drift')
    else:
        atomic_json(manifest, bound)
    library = root / 'geometry.dylib'
    command = ['cc', '-O3', '-std=c11', '-shared', '-fPIC', str(REPO / 'experiments/ddm_rlc1_geometry.c'), '-o', str(library)]
    subprocess.run(command, check=True)
    os.environ['RLC1_GEOMETRY_LIBRARY'] = str(library)
    field = np.memmap(FIELD, mode='r', dtype=np.uint8, shape=(600, 384, 512))
    for frame in frames:
        receipt_path = root / f'frame_{frame:03d}.json'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if not receipt['equal']:
                raise ValueError('prior mismatch needs adjudication')
            for row in receipt['arrays']:
                if sha(Path(row['path'])) != row['sha256']:
                    raise ValueError('retained parity array drift')
            continue
        start = time.monotonic()
        previous = None if frame == 0 else field[frame - 1]
        native, reference = LaneGeometry(previous, config), ReferenceGeometry(previous, config)
        native_bins, reference_bins = np.empty((384, 512), np.uint8), np.empty((384, 512), np.uint8)
        for positions in POSITIONS:
            native_bins.ravel()[positions] = native.contexts(positions)
            reference_bins.ravel()[positions] = reference.contexts(positions)
            symbols = field[frame].ravel()[positions]
            native.observe(positions, symbols)
            reference.observe(positions, symbols)
        rows = []
        for label, bins in [('native', native_bins), ('reference', reference_bins)]:
            path = root / f'frame_{frame:03d}_{label}.u8'
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(bins.tobytes())
            temporary.replace(path)
            rows.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)})
        mismatches = int(np.count_nonzero(native_bins != reference_bins))
        receipt = {'frame': frame, 'positions': native_bins.size, 'equal': mismatches == 0, 'mismatches': mismatches, 'wall_s': time.monotonic() - start, 'arrays': rows, 'concurrency': {'control_processes': 1, 'operator_note': args.concurrency_note}, 'score_claim': False, 'axis': '[macOS-CPU advisory / geometry parity only]'}
        atomic_json(receipt_path, receipt)
        print(json.dumps(receipt), flush=True)
        if mismatches:
            raise ValueError('native/reference geometry mismatch')
    atomic_json(root / 'complete.json', {'complete': True, 'frames': frames, 'position_count': len(frames) * 384 * 512, 'score_claim': False, 'build_command': command, 'library_sha256': sha(library)})


if __name__ == '__main__':
    main()
