"""Retained n600 conditional-residual diagnostic; not an actual receiver codec.

The measured conditioning maps are cached scorer outputs. The pose/calibration
are separately priced historical payloads, NOT transmitted by move40. The
covering classes also grant a source-specific skeleton. No pointwise, marginal
tail, physical source-law, or score claim is made.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import struct
import subprocess
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'

import ddm_eb1_entropy_bound as base
import numpy as np
from scipy import ndimage

ROOT = Path('/Volumes/APDataStore/pact/ddm_eb2_conditional_bound')
REPO = Path(__file__).resolve().parents[1]
SOURCE = Path('/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price')
GT = Path('/Volumes/APDataStore/pact/ddm_eb1_entropy_bound/retained/gt_dali_labels.npy')
ARGMAX = SOURCE / 'seg_final/argmax_n600.npy'
TOKENS = SOURCE / 'parseback/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8'
POSE = ROOT / 'inputs/pose_targets_n600_f16.bin'
CALIBRATION = ROOT / 'inputs/warp_calibration_f64.bin'
ARCHIVE = SOURCE / 'candidate/candidate_runtime/archive.zip'
ARCHIVE_SHA = '986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857'
AXIS = '[macOS-CPU advisory; cached n600 partitions; scorer-free exact bytes]'
N, H, W = 600, 384, 512
CARDINAL = ((-1, 0), (0, -1), (0, 1), (1, 0))
base.ROOT = ROOT
base.Q = 6  # residual alphabet: 0=match, c+1=replace with class c


def put(path, payload):
    """Atomically retain immutable payloads, including nonwinning candidates."""
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('write outside owned store')
    base.storage(path, len(payload))
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'immutable collision: {path}')
    else:
        tmp = path.with_suffix(path.suffix + '.partial')
        with tmp.open('wb') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    return base.fact(path)


def record(path, obj):
    return put(path, (json.dumps(obj, sort_keys=True, indent=2) + '\n').encode())


def load_lie():
    spec = importlib.util.spec_from_file_location('eb2_numpy_lie', REPO / 'src/tac/lie/_se3_numpy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def warp(previous, pose, calibration, lie):
    """Historical XI1 class composite, with every fitted value in a paid input.

    PoseNet-to-motion units and the intra-pair/inter-pair association are
    hypotheses inherited from XI1, not physical SE(3) measurements.
    """
    st, sr, pitch = calibration
    rotation = lie.exp_so3(sr * pose[3:6])
    translation = st * pose[[2, 1, 0]]
    intrinsics = np.array([[910 * W / 1164, 0, W / 2], [0, 910 * H / 874, H / 2], [0, 0, 1.]])
    normal = np.array([0., -np.cos(pitch), -np.sin(pitch)])
    yy, xx = np.indices((H, W))
    grid = np.stack((xx.ravel(), yy.ravel(), np.ones(H * W)))

    def sample(matrix):
        inverse = np.linalg.inv(matrix)
        if not np.isfinite(inverse).all():
            raise ValueError('nonfinite homography inverse')
        coords = np.einsum('ij,jk->ik', inverse, grid, optimize=False)
        with np.errstate(divide='ignore', invalid='ignore'):
            x, y = coords[0] / coords[2], coords[1] / coords[2]
        valid = np.isfinite(x) & np.isfinite(y) & (coords[2] > 0)
        valid &= (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
        ix = np.rint(np.clip(np.nan_to_num(x), 0, W - 1)).astype(np.int64)
        iy = np.rint(np.clip(np.nan_to_num(y), 0, H - 1)).astype(np.int64)
        return previous[iy, ix].reshape(H, W), valid.reshape(H, W)

    ki = np.linalg.inv(intrinsics)
    ground, gv = sample(intrinsics @ (rotation - np.outer(translation, normal) / 1.22) @ ki)
    sky, sv = sample(intrinsics @ rotation @ ki)
    take_ground = gv & np.isin(ground, (0, 1, 3))
    result = np.where(take_ground, ground, previous)
    result = np.where(sv & (sky == 2) & ~take_ground, sky, result)
    result = np.where(previous == 4, previous, result).astype(np.uint8)
    return result, int((~gv).sum()), int((~sv).sum())


def boundary(plane):
    result = np.zeros((H, W), bool)
    hor = plane[:, 1:] != plane[:, :-1]
    vert = plane[1:] != plane[:-1]
    result[:, 1:] |= hor
    result[:, :-1] |= hor
    result[1:] |= vert
    result[:-1] |= vert
    return result


def profile_classes(residual, predictor):
    """Exact C/N orbits preserving residual geometry and predictor-label strata."""
    yy, xx = np.indices((H - 4, W - 4))
    yy, xx = yy + 2, xx + 2
    choose = (yy + 2 * xx) % 5 == 0
    ys, xs = yy[choose], xx[choose]
    collars = np.stack([residual[ys + dy, xs + dx] for dy, dx in base.OFFSETS], axis=1)
    centers = residual[ys, xs]
    potential = np.any(collars != centers[:, None], axis=1)
    groups = defaultdict(list)
    for y, x, collar, center in zip(ys[potential], xs[potential], collars[potential], centers[potential], strict=True):
        signature = tuple(map(int, collar))
        good = base.simple_labels(signature)
        if int(center) not in good or len(good) < 2:
            continue
        around = dict(zip(base.OFFSETS, signature, strict=True))
        neighbors, unsaturated = Counter(around[d] for d in CARDINAL), Counter()
        for dy, dx in CARDINAL:
            color = around[dy, dx]
            other = [(dy + ey, dx + ex) for ey, ex in CARDINAL if (dy + ey, dx + ex) != (0, 0)]
            if all(around[p] == color for p in other):
                unsaturated[color] += 1
        key = (int(predictor[y, x]), int(y) // 64,
               *(neighbors[c] for c in range(6)), *(unsaturated[c] for c in range(6)),
               sum(1 << c for c in good))
        groups[key].append((int(y), int(x)))
    natural, conservative = residual.copy(), residual.copy()
    rows = []
    for key, positions in sorted(groups.items()):
        values = [int(residual[y, x]) for y, x in positions]
        counts = Counter(values)
        if len(counts) < 2:
            continue
        assert key[0] + 1 not in counts  # impossible residual replacement excluded
        remaining, ways = len(values), 1
        for c in sorted(counts):
            ways *= math.comb(remaining, counts[c])
            remaining -= counts[c]
        for p, c in zip(positions, values[1:] + values[:1], strict=True):
            natural[p] = c
        buckets = {c: [p for p, v in zip(positions, values, strict=True) if v == c] for c in counts}
        pairs = []
        while sum(bool(v) for v in buckets.values()) >= 2:
            a, b = sorted((c for c in buckets if buckets[c]), key=lambda c: (-len(buckets[c]), c))[:2]
            p, q = buckets[a].pop(), buckets[b].pop()
            pairs.append([p[0] * W + p[1], q[0] * W + q[1]])
            conservative[p], conservative[q] = residual[q], residual[p]
        assert len(pairs) == min(len(values) // 2, len(values) - max(counts.values()))
        rows.append({'key':list(key), 'positions':[y * W + x for y, x in positions],
                     'counts':dict(counts), 'pairs':pairs, 'cardinality_hex':hex(ways)})
    return rows, natural, conservative


def varint(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def run_encode(mask):
    """Binary alternating run lengths; known frame size and first bit=0."""
    flat = mask.ravel()
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    runs = np.diff(np.r_[0, changes, flat.size])
    if flat[0]:
        runs = np.r_[0, runs]
    return b''.join(varint(int(v)) for v in runs)


def run_decode(payload, offset):
    flat, cursor, bit = np.empty(H * W, bool), 0, False
    while cursor < flat.size:
        value, shift = 0, 0
        while True:
            if offset >= len(payload) or shift > 28:
                raise ValueError('truncated or oversized run')
            byte = payload[offset]
            offset += 1
            value |= (byte & 127) << shift
            if byte < 128:
                break
            shift += 7
        if value == 0 and cursor != 0:
            raise ValueError('noncanonical zero run')
        if cursor + value > flat.size:
            raise ValueError('run exceeds frame')
        flat[cursor:cursor + value] = bit
        cursor += value
        bit = not bit
    return flat.reshape(H, W), offset


def conditional_partition_description_bound_v1(rows, distortion):
    """C/N minimax converse with fixed predictor AND granted residual skeleton."""
    cardinality = math.prod(int(r['cardinality_hex'], 16) for r in rows)
    switches = sum(len(r['pairs']) for r in rows)
    mutable = sum(len(r['positions']) for r in rows)
    qmax = max((len(r['counts']) for r in rows), default=1)
    result = base.partition_description_rate_distortion_lower_bound_v1(cardinality, mutable, qmax, switches, distortion)
    result['natural_minimax_lower_bits_certified'] = max(result['natural_lower_bits_certified'], result['conservative_lower_bits_certified'])
    result.update(switches=switches, mutable_sites=mutable, groups=len(rows), qmax=qmax, cardinality_hex=hex(cardinality))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', type=Path, required=True)
    args = parser.parse_args()
    if args.resume_from.resolve() != (ROOT / 'frames').resolve():
        raise ValueError('resume directory must be owned frame store')
    pins = {str(p):base.fact(p) for p in (GT, ARGMAX, TOKENS, POSE, CALIBRATION, ARCHIVE, Path(__file__), Path(base.__file__), REPO / 'src/tac/lie/_se3_numpy.py')}
    assert pins[str(ARCHIVE)]['sha256'] == ARCHIVE_SHA
    assert pins[str(ARGMAX)]['sha256'] == '50abe278279af3909baf1307389037b5a102250a2d58580e60c9f4135a927667'
    assert pins[str(TOKENS)]['sha256'] == 'b50da438e65b62d5d6f4ca1e151463d097feafd102bbd11d3e0556f849fa4ab5'
    gt_binding = json.loads((GT.parent / 'ARRAY_INPUTS.json').read_text())['gt']
    assert pins[str(GT)] == gt_binding
    record(ROOT / 'INPUTS.json', {'pins':pins, 'axis':AXIS, 'seed':20260910,
        'score_claim':False, 'receiver_valid':False, 'pose_bytes_extra':POSE.stat().st_size + CALIBRATION.stat().st_size,
        'side_information':'cached scored previous partition granted per frame; pose+calibration separately charged; first frame zero predictor',
        'warp':'historical XI1 composite, calibration fixed before this experiment, no new fitting',
        'argv': ['--resume-from', str(args.resume_from)], 'python':platform.python_version(), 'numpy':np.__version__,
        'zlib':zlib.ZLIB_RUNTIME_VERSION, 'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    gt = np.load(GT, mmap_mode='r', allow_pickle=False)
    argmax = np.load(ARGMAX, mmap_mode='r', allow_pickle=False)
    tokens = np.memmap(TOKENS, dtype=np.uint8, mode='r', shape=(N,H,W))
    assert gt.shape == argmax.shape == (N,H,W) and gt.dtype == argmax.dtype == np.uint8
    poses = np.frombuffer(POSE.read_bytes(), '<f2').astype(np.float64).reshape(N,6)
    calibration = np.frombuffer(CALIBRATION.read_bytes(), '<f8')
    assert calibration.shape == (3,) and np.isfinite(poses).all() and np.isfinite(calibration).all()
    lie = load_lie()
    zero = np.zeros((H,W),np.uint8)
    assert np.array_equal(warp(argmax[1], np.zeros(6), calibration, lie)[0], argmax[1])
    started = time.monotonic()
    for t in range(N):
        path = args.resume_from / f'{t:03d}.json'
        if path.exists():
            old = json.loads(path.read_text())
            for artifact in old['artifacts']:
                assert base.fact(artifact['path']) == artifact
            continue
        previous = argmax[t-1] if t else zero
        predicted, bad_ground, bad_sky = warp(previous, poses[t], calibration, lie) if t else (zero,0,0)
        cases = {'persist':previous, 'warp':predicted, 'from_scratch':zero}
        row = {'frame':t, 'artifacts':[], 'cases':{}, 'invalid_ground':bad_ground, 'invalid_rotation':bad_sky,
               'own_argmax_gt_mismatches':int(np.count_nonzero(argmax[t] != gt[t])),
               'own_tokens_gt_mismatches':int(np.count_nonzero(tokens[t] != gt[t]))}
        target_band = boundary(gt[t])
        band3 = ndimage.binary_dilation(target_band, iterations=3)
        for name,pred in cases.items():
            residual = np.where(pred == gt[t], 0, gt[t]+1).astype(np.uint8)
            mismatches = residual > 0
            geom = base.geometry(residual)
            confusion = np.bincount((pred.astype(np.int64)*5+gt[t]).ravel(),minlength=25).reshape(5,5)
            row['cases'][name] = {'geometry':geom, 'disagreement':int(mismatches.sum()),
                'band_rows_128_319':int(mismatches[128:320].sum()),
                'gt_boundary_r0':int((mismatches & target_band).sum()),
                'gt_boundary_r3':int((mismatches & band3).sum()),
                'confusion_pred_rows_gt_cols':confusion.tolist()}
            for c in range(5):
                raw = run_encode(residual == c+1)
                decoded, offset = run_decode(raw,0)
                assert offset == len(raw) and np.array_equal(decoded,residual==c+1)
                row['artifacts'].append(put(ROOT/'rle'/name/f'{t:03d}_{c}.bin',raw))
            if name == 'warp':
                for tag,array in [('predictor',pred),('residual',residual)]:
                    target = ROOT/'retained'/tag/f'{t:03d}.npy'
                    base.persist_array(target,array)
                    row['artifacts'].append(base.fact(target))
                groups,natural,conservative = profile_classes(residual,pred)
                row['groups'] = groups
                for tag, alternate in [('natural',natural),('conservative',conservative)]:
                    target = ROOT/'retained'/tag/f'{t:03d}.npy'
                    base.persist_array(target,alternate)
                    row['artifacts'].append(base.fact(target))
                    check = base.geometry(alternate)
                    for k in ('area','boundary_cells','boundary_edges','pair_edges','components'):
                        assert check[k] == geom[k], (t,tag,k)
                    assert not np.any(alternate == pred+1)
        record(path,row)
        if (t+1)%20 == 0:
            print(json.dumps({'frames_completed':t+1,'seconds':time.monotonic()-started}),flush=True)
    rows = [json.loads((args.resume_from/f'{t:03d}.json').read_text()) for t in range(N)]
    results = {'axis':AXIS,'n':N,'cells':N*H*W,'selection_mode':'all_scored_pairs_0_through_599',
        'score_claim':False,'receiver_valid':False,'pose_and_calibration_extra_bytes':POSE.stat().st_size+CALIBRATION.stat().st_size,
        'own_argmax_gt_mismatches':sum(r['own_argmax_gt_mismatches'] for r in rows),
        'own_tokens_gt_mismatches':sum(r['own_tokens_gt_mismatches'] for r in rows),'cases':{}}
    for name in ('persist','warp','from_scratch'):
        codebytes, per_class, decoded_masks = 0, [], []
        for c in range(5):
            raw = b''.join((ROOT/'rle'/name/f'{t:03d}_{c}.bin').read_bytes() for t in range(N))
            put(ROOT/'retained'/f'{name}_{c}.rle',raw)
            # The complete framed payloads from two independent compressor calls survive.
            header = struct.pack('<4sBHHH',b'ERL1',c,N,H,W)
            a = header + zlib.compress(raw,9)
            b = header + zlib.compress(raw,9)
            facts = [put(ROOT/'retained'/f'{name}_{c}{suffix}.erl',payload) for suffix,payload in (('',a),('.repeat',b))]
            assert a == b
            decoded_raw = zlib.decompress(a[len(header):])
            assert decoded_raw == raw
            offset,masks = 0,[]
            for _t in range(N):
                mask, offset = run_decode(decoded_raw,offset)
                masks.append(mask)
            assert offset == len(decoded_raw)
            decoded_masks.append(masks)
            codebytes += len(a)
            per_class.append({'class':c,'bytes':len(a),'raw_rle_bytes':len(raw),'artifacts':facts})
        # Full joint residual parse-back, using only the declared conditional predictor.
        for t in range(N):
            previous = argmax[t-1] if t else zero
            pred = (warp(previous,poses[t],calibration,lie)[0] if t else zero) if name=='warp' else (previous if name=='persist' else zero)
            recovered = pred.copy()
            coverage = np.zeros((H,W),np.uint8)
            for c in range(5):
                mask = decoded_masks[c][t]
                coverage += mask
                recovered[mask] = c
            assert coverage.max() <= 1 and np.array_equal(recovered,gt[t])
        aggregate = {k:sum(r['cases'][name][k] for r in rows) for k in ('disagreement','band_rows_128_319','gt_boundary_r0','gt_boundary_r3')}
        aggregate['per_class'] = per_class
        aggregate['coder_bytes'] = codebytes
        aggregate['boundary_edges'] = sum(r['cases'][name]['geometry']['boundary_edges'] for r in rows)
        aggregate['components_by_residual_symbol'] = np.sum([r['cases'][name]['geometry']['components'] for r in rows],axis=0).tolist()
        aggregate['confusion_pred_rows_gt_cols'] = np.sum([r['cases'][name]['confusion_pred_rows_gt_cols'] for r in rows],axis=0).tolist()
        aggregate['first_frame_coder_raw_rle_bytes'] = sum((ROOT/'rle'/name/f'000_{c}.bin').stat().st_size for c in range(5))
        aggregate['transition_disagreement_excluding_bootstrap'] = aggregate['disagreement']-rows[0]['cases'][name]['disagreement']
        results['cases'][name] = aggregate
        del decoded_masks
    all_groups = [g for row in rows for g in row['groups']]
    results['bounds'] = [conditional_partition_description_bound_v1(all_groups,d) for d in (6270,12540,25080)]
    results['per_class_bounds_D12540'] = [conditional_partition_description_bound_v1([g for g in all_groups if set(map(int,g['counts']))=={0,c+1}],12540) for c in range(5)]
    results['all_600_invariants_and_joint_coder_parseback'] = True
    record(ROOT/'RESULT.json',results)
    print(json.dumps({'complete':True,'sizes':{k:v['coder_bytes'] for k,v in results['cases'].items()}}),flush=True)


if __name__ == '__main__':
    main()
