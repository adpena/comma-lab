#!/usr/bin/env python3
"""RE-VERIFY a composed field's seg repairs, per pair, on the composed renders.

Two arms' token edits compose at the token level whenever no cell is written twice, and
that is checkable by intersection. Their SEG effects do not: this arm's repairs were
realized against ONE base's renders, and a sister arm's edits move those renders on every
pair the two share. So the composed field's repaired-cell count must be RECOUNTED, never
carried across (MAIN 2026-09-10: "recompute, do not assume").

What this measures, for each pair given: the realized argmax of the pair's plane in the
BASE field and in the COMPOSED field, against the DALI GT, through the receiver's own
`render_frame1` at batch 1 and the frozen CPU-torch SegNet — the same instrument every
other seg row on this arm uses. It reports per pair the flips before, the flips after, and
therefore the repairs THIS composition actually delivers on THIS base.

No scorer weights ship, no archive is built, nothing is promoted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'experiments'), str(REPO / 'src')]
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import ddm_jg1_seg_solve as jg1  # noqa: E402
import ddm_sj1_multipass_token_predistortion as sj1  # noqa: E402
import ddm_up2_shipping_pose_solve as up2  # noqa: E402

AXIS = '[macOS-CPU advisory, jg1 instrument, DALI GT lineage]'


def load_field(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {str(i) for i in range(sj1.N_PAIRS)}:
            raise SystemExit(f'{path} is not a {sj1.N_PAIRS}-plane carry field')
        return np.stack([data[str(i)] for i in range(sj1.N_PAIRS)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-field', type=Path, required=True,
                        help='the field the composition is built ON (the live pointer ships it)')
    parser.add_argument('--composed-field', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, required=True,
                        help='json list of the pairs this arm edited; only these can differ')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    import torch
    torch.set_num_threads(args.threads)

    base = load_field(args.base_field)
    comp = load_field(args.composed_field)
    pairs = sorted(int(p) for p in json.loads(args.pairs.read_text()))

    # Fail closed: a pair outside the declared list must be byte-identical in the two
    # fields, or the "only these pairs can differ" premise is false and every count below
    # would be measured on the wrong object.
    differing = np.flatnonzero((base != comp).reshape(sj1.N_PAIRS, -1).any(axis=1))
    unexpected = sorted(set(int(p) for p in differing) - set(pairs))
    if unexpected:
        raise SystemExit(f'{len(unexpected)} pairs differ outside the declared set: {unexpected[:8]}')

    gt = jg1.load_gt_seg_labels(up2.LINEAGE_DALI)
    body = sj1.load_body(with_raw=False)
    rows = []
    started = time.time()
    for i, pair in enumerate(pairs):
        before = int((sj1.argmax_for_tokens(body, base[pair], pair) != gt[pair]).sum())
        after = int((sj1.argmax_for_tokens(body, comp[pair], pair) != gt[pair]).sum())
        rows.append(dict(pair=pair, flips_before=before, flips_after=after,
                         repaired=before - after,
                         tokens_changed=int((base[pair] != comp[pair]).sum())))
        if (i + 1) % 10 == 0:
            print(json.dumps(dict(done=i + 1, of=len(pairs),
                                  seconds=round(time.time() - started, 1))), flush=True)

    repaired = sum(r['repaired'] for r in rows)
    result = dict(
        schema='ddm_sj1_compose_segcheck.v1', axis=AXIS, score_claim=False,
        base_field=str(args.base_field), composed_field=str(args.composed_field),
        pairs=len(pairs), tokens_changed=sum(r['tokens_changed'] for r in rows),
        repaired_on_this_base=repaired,
        pairs_that_lost_a_repair=[r['pair'] for r in rows if r['repaired'] < 0],
        pairs_with_zero_net=[r['pair'] for r in rows if r['repaired'] == 0],
        elapsed_seconds=time.time() - started, rows=rows,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
