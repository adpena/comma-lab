#!/usr/bin/env python3
"""Census the pass-5 split of the shipped seg residual: repaired vs remaining.

The pass-5 search repaired 237 of the 12,614 cells the move-37 body ships and left
12,377.  Whether or not pass 5 admits, those two sets are what the representation-level
boundary charter inherits, and the interesting question is not their sizes -- it is
whether the 237 the last token move can still reach differ IN KIND from the 12,377 it
cannot.  If they do not, the residual is homogeneous and the token grid has simply run
out of budget; if they do, the difference names the sub-family a successor should target.

Inputs, all already on disk, all MEASURED:
  * the SHIPPED parse-back argmax of the move-35/37 body (`pass4/seg_final/argmax_n600.npy`)
  * the DALI GT label table (`ddm_jg1_seg_solve.load_gt_seg_labels`)
  * the pass-5 accepted-move rows, which carry each repaired cell's (site_y, site_x)

Facets, matching the memo's Sections 16 and 21 so the tables are comparable across
passes: GT class, row band, 4-connected component size within the residual, and the
per-pair spread.  Nothing here is projected and no scorer runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'experiments'), str(REPO / 'src')]
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import ddm_jg1_seg_solve as jg1  # noqa: E402
import ddm_up2_shipping_pose_solve as up2  # noqa: E402

N_PAIRS, EVAL_H, EVAL_W = 600, 384, 512
#: comma10k canonical order.  NEVER luma-sorted -- the sort gives a different, wrong map.
CLASS_NAMES = ('Road', 'Lane', 'Undrivable', 'Movable', 'MyCar')
ROW_BANDS = ((0, 128), (128, 192), (192, 256), (256, 320), (320, 384))
AXIS = '[macOS-CPU advisory, jg1 instrument, DALI GT lineage]'


def accepted_sites(pass_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Return a (600, H, W) bool mask of cells the pass-5 accepted moves repaired."""
    mask = np.zeros((N_PAIRS, EVAL_H, EVAL_W), dtype=bool)
    rows = []
    for shard in sorted(pass_dir.glob('rows_shard_*.jsonl')):
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for move in row['accepted']:
                # `repaired` counts the cells the move fixed; the row records the SITE it
                # was aimed at.  A move that repaired two cells names only one of them, so
                # the site mask is a LOWER bound on the repaired set and is labelled as one.
                mask[int(row['pair']), int(move['site_y']), int(move['site_x'])] = True
                rows.append(dict(pair=int(row['pair']), **move))
    return mask, rows


def components(residual: np.ndarray) -> np.ndarray:
    """4-connected component size for every residual cell, computed per pair.

    A tiny union-find over the residual cells only: the sets are ~12k cells in 600 planes,
    so a full-frame labelling would be 118 M cells of wasted work.
    """
    sizes = np.zeros(residual.shape, dtype=np.int32)
    for pair in range(N_PAIRS):
        ys, xs = np.nonzero(residual[pair])
        if not len(ys):
            continue
        index = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
        parent = list(range(len(ys)))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for (y, x), i in index.items():
            for dy, dx in ((-1, 0), (0, -1)):
                j = index.get((y + dy, x + dx))
                if j is not None:
                    ra, rb = find(i), find(j)
                    if ra != rb:
                        parent[ra] = rb
        counts: dict[int, int] = {}
        for i in range(len(ys)):
            root = find(i)
            counts[root] = counts.get(root, 0) + 1
        for i in range(len(ys)):
            sizes[pair, ys[i], xs[i]] = counts[find(i)]
    return sizes


def facet_table(mask: np.ndarray, gt: np.ndarray, sizes: np.ndarray) -> dict:
    total = int(mask.sum())
    if total == 0:
        return dict(cells=0)
    classes = gt[mask]
    rows = np.nonzero(mask)[1]
    comp = sizes[mask]
    per_pair = mask.reshape(N_PAIRS, -1).sum(axis=1)
    area = np.bincount(gt.reshape(-1), minlength=len(CLASS_NAMES)) / gt.size
    by_class = {}
    for k, name in enumerate(CLASS_NAMES):
        cells = int((classes == k).sum())
        by_class[name] = dict(cells=cells, share=cells / total,
                              frame_area=float(area[k]),
                              over_representation=(cells / total) / float(area[k]) if area[k] else None)
    by_band = {f'{lo}-{hi - 1}': float(((rows >= lo) & (rows < hi)).mean())
               for lo, hi in ROW_BANDS}
    by_size = {}
    for label, lo, hi in (('1', 1, 1), ('2', 2, 2), ('3-4', 3, 4), ('5-9', 5, 9),
                          ('10-24', 10, 24), ('>=25', 25, 1 << 30)):
        cells = int(((comp >= lo) & (comp <= hi)).sum())
        by_size[label] = dict(cells=cells, share=cells / total)
    # BOTH singleton statistics, because the memo's Section 21 compared one against the
    # other across two passes and read a trend out of the level switch.  A component of
    # size k contributes k cells, so the component count is sum(1/k) over cells.
    n_components = float((1.0 / comp).sum())
    return dict(
        cells=total,
        row_centroid=float(rows.mean()),
        components=n_components,
        cells_per_component=total / n_components,
        mean_component_size_over_cells=float(comp.mean()),
        singleton_share_of_cells=float((comp == 1).mean()),
        singleton_share_of_components=float((comp == 1).sum()) / n_components,
        by_class=by_class, by_row_band=by_band, by_component_size=by_size,
        per_pair=dict(mean=float(per_pair.mean()), median=float(np.median(per_pair)),
                      max=int(per_pair.max()), min=int(per_pair.min()),
                      pairs_clean=int((per_pair == 0).sum()),
                      worst_100_share=float(np.sort(per_pair)[-100:].sum() / total)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--argmax', type=Path, required=True,
                        help='shipped parse-back argmax of the live body, (600,H,W) uint8')
    parser.add_argument('--pass-dir', type=Path, required=True, help='pass-5 shard rows')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    argmax = np.load(args.argmax)
    if argmax.shape != (N_PAIRS, EVAL_H, EVAL_W):
        raise SystemExit(f'argmax has shape {argmax.shape}')
    gt = jg1.load_gt_seg_labels(up2.LINEAGE_DALI)
    if gt.shape != argmax.shape:
        raise SystemExit(f'GT has shape {gt.shape}, argmax {argmax.shape}')

    residual = argmax != gt
    repaired, rows = accepted_sites(args.pass_dir)
    # Every site a move repaired must actually BE a residual cell of the shipped body;
    # if it is not, the pass and the parse-back are describing different objects.
    stray = int((repaired & ~residual).sum())
    sizes = components(residual)
    remaining = residual & ~repaired

    result = dict(
        schema='ddm_sj1_pass5_census.v1', axis=AXIS, score_claim=False,
        argmax=str(args.argmax), pass_dir=str(args.pass_dir),
        residual_cells=int(residual.sum()),
        repaired_site_cells=int(repaired.sum()),
        remaining_cells=int(remaining.sum()),
        accepted_moves=len(rows),
        repaired_sites_not_in_shipped_residual=stray,
        residual=facet_table(residual, gt, sizes),
        repaired=facet_table(repaired, gt, sizes),
        remaining=facet_table(remaining, gt, sizes),
        note=('repaired_site_cells counts one site per accepted move, so it is a LOWER '
              'bound on the 237 cells PASS_RESULT reports; two moves repaired two cells '
              'from one site.  The kind-comparison between the two sets is unaffected.'),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('residual', 'repaired', 'remaining')}, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
