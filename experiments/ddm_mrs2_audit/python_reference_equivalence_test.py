"""Run or resume the retained 48-pair C/Python proof from the repository root."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, str(repo / 'experiments/ddm_mrs2_profile.py'),
                    '--native', '--resume-from', str(args.resume_from)], cwd=repo, check=True)
    candidate = json.loads((args.resume_from / 'RESULT.json').read_text())
    reference = json.loads(Path('/Volumes/APDataStore/pact/ddm_mrs2/profile_python/RESULT.json').read_text())
    expected = {row['frame']: row for row in reference['rows']}
    assert candidate['matched_pairs'] == len(candidate['rows']) == len(expected) == 48
    assert {row['frame'] for row in candidate['rows']} == set(expected)
    for row in candidate['rows']:
        for key in ('token', 'raw'):
            artifact = row[key]
            path = Path(artifact['path'])
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            assert path.stat().st_size == artifact['bytes']
            assert digest == artifact['sha256'] == expected[row['frame']][key]['sha256']
    print('48/48 same-state C/Python token and raw vectors match.')


if __name__ == '__main__':
    main()
