#!/usr/bin/env python3
"""Price sj1's pass-5 token field by REAL re-encode under the LIVE tail coder.

The predecessor script `experiments/ddm_sj1_successor_price.sh` encodes through the
cl2 HPAC receiver copy.  That coder was replaced under the arm: pointer move 36
shipped tc1's 35-weight shared mixer over HPAC, and move 37 (cmp2) kept it.  A byte
delta measured on a coder the archive no longer uses is not this candidate's price,
so this module re-derives the number on the coder that is actually live.

The encode loop is `ddm_cmp1_compose.encode`'s, re-pointed at the live cmp2 tree and
parameterised by the field.  Both streams are emitted per run: `control` is the
original FreeCorrector/HPAC coder (a free cross-check, since its rows are the mixer's
input anyway) and `mixed` is tc1's mixer -- the shipped one.

Gates, in the order they must pass before any byte number is quoted:

1. the live tree's own reader returns the tail stream this module targets;
2. the CONTROL field (the shipped pass-4 admitted field) re-encodes to a `mixed`
   stream byte-identical to that shipped tail;
3. the two independent encodes of each field agree byte-for-byte;
4. the candidate's price is the EXACT container delta -- `pack_archive` on the real
   member -- never a first-order -log2 p sum (rp1 measured 0.1445 realisation on the
   saving side; sj1 measured 1.279x on the cost side).

Per-frame ideal bits are retained for both coders so a Lagrange subset can be RANKED
per pair; a selected subset is then priced by its own real re-encode, because the
per-pair ledger sum under-charges (pass 3 measured +19.6 B on a 370-pair subset).

No scorer, no dispatch, no pointer write.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
sys.dont_write_bytecode = True
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np  # noqa: E402
from experiments import ddm_jg2_tail_reencode as jg2  # noqa: E402
from experiments import ddm_tc1_mixer_codec as tc1  # noqa: E402

ROOT = Path('/Volumes/VertigoDataTier/pact/ddm_sj1_pass5_price')
BULK = Path('/Volumes/APDataStore/pact/ddm_sj1_pass5')
LIVE = Path('/Volumes/VertigoDataTier/pact/ddm_cmp2_compose/candidate_runtime')
SJ1 = Path('/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion')
CONTROL_NPZ = SJ1 / 'admission_pass4/field_admitted.npz'
CANDIDATE_NPZ = SJ1 / 'passes/pass5_gt/field_after.npz'
TC1_WEIGHTS = Path('/Volumes/VertigoDataTier/pact/ddm_tc1_tail_shared_mixer/rebase_pc2/mixer/weights_i8.bin')
POINTER = REPO / '.omx/state/canonical_frontier_pointer.json'
AXIS = '[macOS-CPU advisory / scorer-free EXACT byte measurement]'
TAIL_PREFIX_BYTES = 96
FIELDS = ('control', 'candidate')
N_PAIRS, EVAL_H, EVAL_W = 600, 384, 512


# ----------------------------------------------------------------------------------
# custody helpers
# ----------------------------------------------------------------------------------

def preflight(path: Path, need: int = 1 << 20) -> None:
    """Refuse a write outside the owned stores, or one the volume cannot hold."""
    path = Path(path).resolve()
    root = ROOT if path.is_relative_to(ROOT) else BULK if path.is_relative_to(BULK) else None
    if root is None:
        raise ValueError(f'write outside owned stores: {path}')
    mount = Path('/Volumes') / path.parts[2]
    if shutil.disk_usage(mount).free < need + 2 * 1024 ** 3:
        raise RuntimeError(f'storage preflight refused on {mount}: need {need} B')
    path.parent.mkdir(parents=True, exist_ok=True)


def write(path: Path, payload: bytes) -> dict:
    path = Path(path)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'immutable payload changed: {path}')
        return jg2.file_fact(path)
    preflight(path, len(payload))
    jg2.atomic_write(path, payload)
    return jg2.file_fact(path)


def record(path: Path, value: object) -> object:
    preflight(path)
    jg2.atomic_json(path, value)
    return value


def runtime_facts(root: Path) -> dict:
    facts = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            facts[str(path.relative_to(root))] = jg2.file_fact(path)
    return facts


# ----------------------------------------------------------------------------------
# init -- pin the pointer, the tree, the rider and both fields
# ----------------------------------------------------------------------------------

def split_tail(tail: bytes) -> tuple[bytes, bytes, bytes]:
    """Return (prefix, weights, stream) for the shipped TC1M tail layout."""
    prefix = tail[:TAIL_PREFIX_BYTES]
    rider = tail[TAIL_PREFIX_BYTES:]
    if rider[:len(tc1.MAGIC)] != tc1.MAGIC:
        raise RuntimeError('shipped tail does not carry a raw TC1M rider at the 96 B prefix')
    weights, stream = tc1.unpack_rider(rider)
    return prefix, bytes(weights), stream


def build_tail(prefix: bytes, weights: bytes, stream: bytes) -> bytes:
    return prefix + tc1.MAGIC + weights + stream


def field_to_u8(npz_path: Path, destination: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as data:
        if set(data.files) != {str(i) for i in range(N_PAIRS)}:
            raise ValueError(f'{npz_path} does not carry all {N_PAIRS} planes')
        field = np.stack([data[str(i)] for i in range(N_PAIRS)])
    if field.shape != (N_PAIRS, EVAL_H, EVAL_W) or field.dtype != np.uint8 or field.max() >= tc1.K:
        raise ValueError(f'{npz_path} geometry/alphabet is wrong')
    return write(destination, field.tobytes())


def guard() -> dict:
    inputs = json.loads((ROOT / 'INPUTS.json').read_text())
    live = json.loads(POINTER.read_text())['our_local_frontier_contest_cuda']
    if live['archive_sha256'] != inputs['pointer_archive']['sha256']:
        raise RuntimeError('POINTER_MOVED: refuse; the price must be measured against the live row')
    for relative, fact in inputs['runtime_sources'].items():
        if jg2.file_fact(LIVE / relative) != fact:
            raise RuntimeError(f'live runtime drift at {relative}')
    for name, entry in inputs['fields'].items():
        if jg2.file_fact(Path(entry['u8']['path'])) != entry['u8']:
            raise RuntimeError(f'{name} field custody changed')
    return inputs


def add_field(name: str, npz_path: Path) -> dict:
    """Register a further field for pricing -- the Lagrange-SELECTED subset.

    The subset the sweep picks is a different object from the full pass field and must be
    priced by its OWN real re-encode: the per-pair ledger sum RANKS pairs but under-charges
    the container (pass 3 measured +19.6 B on a 370-pair subset, pass 4 +5.70 B on 112).
    At pass 5's margin that difference is decisive, not cosmetic.
    """
    inputs = guard()
    if name in inputs['fields']:
        raise ValueError(f'field {name} is already registered; fields are immutable here')
    if not name.isidentifier():
        raise ValueError('field name must be a plain identifier')
    u8 = field_to_u8(npz_path, ROOT / 'retained/fields' / f'{name}.u8')
    control = np.fromfile(inputs['fields']['control']['u8']['path'],
                          dtype=np.uint8).reshape(N_PAIRS, EVAL_H, EVAL_W)
    plane = np.fromfile(u8['path'], dtype=np.uint8).reshape(N_PAIRS, EVAL_H, EVAL_W)
    per_pair = (control != plane).reshape(N_PAIRS, -1).sum(axis=1)
    inputs['fields'][name] = dict(
        npz=jg2.file_fact(npz_path), u8=u8,
        delta_vs_control=dict(tokens_changed=int(per_pair.sum()),
                              pairs_changed=int((per_pair > 0).sum()),
                              per_pair=per_pair.astype(int).tolist()))
    return record(ROOT / 'INPUTS.json', inputs)


def initialize() -> dict:
    if (ROOT / 'INPUTS.json').exists():
        return guard()
    pointer = json.loads(POINTER.read_text())
    live = pointer['our_local_frontier_contest_cuda']
    archive = jg2.file_fact(LIVE / 'archive.zip')
    if archive['sha256'] != live['archive_sha256']:
        raise RuntimeError('configured tree is not the live pointer')

    parts = jg2.split_member(jg2.read_archive_member(LIVE / 'archive.zip'))
    prefix, weights, stream = split_tail(parts['tail'])
    if weights != TC1_WEIGHTS.read_bytes():
        raise RuntimeError('shipped rider weights are not tc1 mixer/weights_i8.bin')
    if build_tail(prefix, weights, stream) != parts['tail']:
        raise RuntimeError('tail split is not lossless')

    null = ROOT / 'retained/null_archive.zip'
    preflight(null, archive['bytes'])
    jg2.pack_archive(jg2.join_member(parts), null)
    if null.read_bytes() != (LIVE / 'archive.zip').read_bytes():
        raise RuntimeError('null rebuild differs from the live archive')

    census = {k: write(ROOT / 'retained/source' / (k + '.bin'), v) for k, v in parts.items()}
    fields = {}
    planes = {}
    for name, npz in (('control', CONTROL_NPZ), ('candidate', CANDIDATE_NPZ)):
        u8 = field_to_u8(npz, BULK / 'fields' / f'{name}.u8')
        fields[name] = dict(npz=jg2.file_fact(npz), u8=u8)
        planes[name] = np.fromfile(u8['path'], dtype=np.uint8).reshape(N_PAIRS, EVAL_H, EVAL_W)
    differ = planes['control'] != planes['candidate']
    per_pair = differ.reshape(N_PAIRS, -1).sum(axis=1)
    result = dict(
        schema='ddm_sj1_pass5_price_inputs.v1', axis=AXIS, score_claim=False, seed=20260909,
        pointer=pointer, pointer_archive=archive, live_tree=str(LIVE),
        runtime_sources=runtime_facts(LIVE), census=census,
        tail=dict(bytes=len(parts['tail']), prefix_bytes=len(prefix), weight_bytes=len(weights),
                  stream_bytes=len(stream), stream_sha256=jg2.sha256_bytes(stream),
                  weights_sha256=jg2.sha256_bytes(weights)),
        fields=fields,
        field_delta=dict(tokens_changed=int(per_pair.sum()),
                         pairs_changed=int((per_pair > 0).sum()),
                         per_pair=per_pair.astype(int).tolist()),
        break_even_bits_per_changed_token=10.271545430934173,
        preregistered_band_bits=[7.7203, 13.36],
        admit_bar_net_dS=-2e-05,
        storage_policy='Vertigo receipts + streams; APDataStore fields and checkpoints',
    )
    return record(ROOT / 'INPUTS.json', result)


# ----------------------------------------------------------------------------------
# encode -- ddm_cmp1_compose.encode's loop, re-pointed at the live tree
# ----------------------------------------------------------------------------------

def encode(field_name: str, tag: str, stop: int) -> dict:
    import torch
    inputs = guard()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(inputs['seed'])
    np.random.seed(inputs['seed'])
    random.seed(inputs['seed'])
    torch.use_deterministic_algorithms(True)

    work = ROOT / 'encode' / field_name / tag
    preflight(work / 'build')
    route = jg2.load_route_b()
    library, build = jg2.compile_rc64(work, route, f'sj1p5_{field_name}_{tag}')
    residual, renderer, renderer_dir = jg2.load_runtime(LIVE)
    from runtime.free_corrector import FreeCorrector
    from runtime.hpac_inference import optimize_sparse_evaluator

    parts = residual.read_residual_archive(LIVE / 'archive.zip')
    if jg2.sha256_bytes(parts.token_stream) != inputs['tail']['stream_sha256']:
        raise RuntimeError("the live tree's own reader no longer returns the pinned tail stream")
    target = jg2.load_tokens(Path(inputs['fields'][field_name]['u8']['path']))
    model = renderer.load_hpac(residual.materialize_ihs1(parts.hpac_blob, renderer), torch.device('cpu'))
    sparse = residual._sparse_class(renderer_dir)(model, EVAL_H, EVAL_W)
    corrector, cold = FreeCorrector(EVAL_H * EVAL_W), FreeCorrector(EVAL_H * EVAL_W)
    mixer = tc1.SharedMixer(TC1_WEIGHTS.read_bytes())

    binding = dict(inputs_sha=jg2.sha256_file(ROOT / 'INPUTS.json'),
                   source_sha=jg2.sha256_file(Path(__file__)),
                   codec_sha=jg2.sha256_file(Path(tc1.__file__)),
                   jg2_sha=jg2.sha256_file(Path(jg2.__file__)),
                   field=inputs['fields'][field_name]['u8']['sha256'],
                   runtime_sources={k: v['sha256'] for k, v in inputs['runtime_sources'].items()},
                   build=build)

    start = 0
    per_frame = np.zeros((2, N_PAIRS), dtype=np.float64)
    latest = work / 'LATEST.json'
    state = None
    if latest.exists():
        receipt = json.loads(latest.read_text())
        if receipt['binding'] != binding or jg2.file_fact(Path(receipt['payload']['path'])) != receipt['payload']:
            raise RuntimeError('checkpoint binding drift')
        with np.load(receipt['payload']['path'], allow_pickle=False) as data:
            state = {k: data[k] for k in data.files}
        start = int(state['frame'][0])
        jg2.load_corrector_state(corrector, {k[2:]: v for k, v in state.items() if k.startswith('c_')})
        mixer.restore({k[2:]: v for k, v in state.items() if k.startswith('m_')})
        if mixer.frame != start:
            raise RuntimeError('mixer checkpoint boundary mismatch')
        per_frame = state['per_frame']
    if not 0 <= start <= stop <= N_PAIRS:
        raise ValueError('invalid frame range')

    encoders = [route.NativeRc64Encoder(library, None if state is None else state[k].tobytes())
                for k in ('control_encoder', 'mixed_encoder')]
    previous = torch.zeros((1, EVAL_H, EVAL_W), dtype=torch.long)
    if start:
        previous[0] = torch.from_numpy(target[start - 1].astype(np.int64))
    groups = [np.flatnonzero(m.cpu().numpy().reshape(-1)) for m in renderer.group_masks(torch.device('cpu'))]
    started = time.monotonic()
    with torch.inference_mode():
        optimize_sparse_evaluator(sparse)
        for frame in range(start, stop):
            previous_cpu = None if frame == 0 else previous[0].numpy().astype(np.uint8)
            boundary = (np.full(EVAL_H * EVAL_W, 4, dtype=np.uint8) if frame == 0
                        else residual._boundary_buckets(previous_cpu).reshape(-1))
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(torch.tensor([frame]), previous)
            corrector.begin_frame(boundary)
            mixer.begin_frame()
            truth = np.asarray(target[frame]).reshape(-1)
            bits = [0., 0.]
            for group, positions in enumerate(groups):
                logits = sparse.selected_logits(current, context, group).cpu().numpy()
                predicted = logits.argmax(axis=1).astype(np.int64)
                feature = boundary[positions].astype(np.int64) * tc1.K + predicted
                probability = residual._probability_table(logits + parts.table.values[feature],
                                                          renderer.HPAC_LOGIT_PRECISION)
                cs = corrector.group_state(probability, predicted, positions)
                original = corrector.coding_row(cs)
                coding = mixer.coding(original, positions, current[0].numpy().astype(np.uint8), previous_cpu)
                symbols = truth[positions].astype(np.int32)
                for i, rows in enumerate((original, coding)):
                    encoders[i].encode(symbols, rows)
                    freq = tc1.frequencies(rows)
                    bits[i] += float(-np.log2(
                        freq[np.arange(len(symbols)), symbols].astype(float) / tc1.TOTAL).sum())
                corrector.observe(cs, symbols.astype(np.int64))
                current.reshape(-1)[torch.from_numpy(positions)] = torch.from_numpy(symbols.astype(np.int64))
            plane = current[0].numpy().astype(np.uint8)
            np.testing.assert_array_equal(plane, target[frame])
            corrector.end_frame(plane.reshape(-1))
            mixer.end_frame(plane, previous_cpu)
            previous = current
            per_frame[:, frame] = bits
            if (frame + 1) % 10 == 0 or frame + 1 == stop:
                captured = jg2.corrector_state(corrector)
                if jg2.uncaptured_divergent_state(corrector, cold, set(captured)):
                    raise RuntimeError('checkpoint would omit mutable corrector state')
                values = {'c_' + k: v for k, v in captured.items()}
                values.update({'m_' + k: v for k, v in mixer.snapshot().items()})
                values.update(frame=np.array([frame + 1]), per_frame=per_frame.copy(),
                              control_encoder=np.frombuffer(encoders[0].snapshot(), dtype=np.uint8),
                              mixed_encoder=np.frombuffer(encoders[1].snapshot(), dtype=np.uint8))
                payload = BULK / 'encode' / field_name / tag / f'stage_{frame + 1:04d}.npz'
                preflight(payload, sum(v.nbytes for v in values.values()))
                if payload.exists():
                    with np.load(payload, allow_pickle=False) as old:
                        if set(old.files) != set(values) or any(
                                not np.array_equal(old[k], v) for k, v in values.items()):
                            raise RuntimeError('immutable checkpoint differs')
                else:
                    temporary = payload.with_suffix('.npz.new')
                    with temporary.open('wb') as handle:
                        np.savez_compressed(handle, **values)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, payload)
                receipt = dict(binding=binding, frame=frame + 1, payload=jg2.file_fact(payload))
                record(work / f'STAGE_{frame + 1:04d}.json', receipt)
                record(latest, receipt)
                print(json.dumps(dict(field=field_name, tag=tag, frame=frame + 1,
                                      seconds=round(time.monotonic() - started, 1))), flush=True)

    outputs = {}
    for name, encoder in zip(('control', 'mixed'), encoders):
        outputs[name + '_envelope'] = write(work / f'{name}_{stop:04d}.envelope', encoder.finish())
        raw = ctypes.string_at(encoder.library.rc64_encoder_data(encoder.context),
                               int(encoder.library.rc64_encoder_size(encoder.context)))
        outputs[name] = write(work / f'{name}_{stop:04d}.rc64', raw)
    identical = (stop == N_PAIRS
                 and Path(outputs['mixed']['path']).read_bytes() == parts.token_stream)
    result = dict(schema='ddm_sj1_pass5_encode.v1', axis=AXIS, score_claim=False,
                  field=field_name, tag=tag, frames=stop, start_frame=start,
                  outputs=outputs, mixed_matches_shipped_tail=identical,
                  ideal_bytes={k: float(per_frame[i].sum() / 8) for i, k in enumerate(('control', 'mixed'))},
                  per_frame_ideal_bits=per_frame.tolist(), binding=binding)
    record(work / f'ENCODE_{stop:04d}.json', result)
    if stop == N_PAIRS and field_name == 'control' and not identical:
        raise RuntimeError('CONTROL IDENTITY FAILED: the shipped field does not re-encode to the shipped tail')
    return result


# ----------------------------------------------------------------------------------
# price -- the exact container delta
# ----------------------------------------------------------------------------------

def load_encode(field_name: str, tag: str) -> dict:
    path = ROOT / 'encode' / field_name / tag / f'ENCODE_{N_PAIRS:04d}.json'
    if not path.exists():
        raise RuntimeError(f'missing completed encode: {path}')
    return json.loads(path.read_text())


def price(tags: tuple[str, ...], candidate: str = 'candidate', seg_gain_bytes: float | None = None,
          seg_cells: float | None = None) -> dict:
    inputs = guard()
    parts = jg2.split_member(jg2.read_archive_member(LIVE / 'archive.zip'))
    prefix, weights, shipped_stream = split_tail(parts['tail'])
    live_bytes = inputs['pointer_archive']['bytes']
    fields = ('control', candidate)

    streams, twins, ideal = {}, {}, {}
    for name in fields:
        runs = [load_encode(name, tag) for tag in tags]
        payloads = [Path(r['outputs']['mixed']['path']).read_bytes() for r in runs]
        if any(p != payloads[0] for p in payloads[1:]):
            raise RuntimeError(f'{name}: independent encodes disagree; the delta would be variance')
        streams[name] = payloads[0]
        twins[name] = [r['outputs']['mixed']['sha256'] for r in runs]
        ideal[name] = dict(mixed=runs[0]['ideal_bytes']['mixed'],
                           control=runs[0]['ideal_bytes']['control'],
                           per_frame=runs[0]['per_frame_ideal_bits'])
    if streams['control'] != shipped_stream:
        raise RuntimeError('CONTROL IDENTITY FAILED at price time')

    archives = {}
    for name in fields:
        member = jg2.join_member(dict(parts, tail=build_tail(prefix, weights, streams[name])))
        destination = ROOT / 'retained' / f'archive_{name}.zip'
        preflight(destination, len(member) + 4096)
        jg2.pack_archive(member, destination)
        archives[name] = jg2.file_fact(destination)
    if archives['control']['bytes'] != live_bytes:
        raise RuntimeError('rebuilt control archive does not reproduce the live byte count')

    entry = inputs['fields'][candidate]
    delta = entry.get('delta_vs_control') or inputs['field_delta']
    changed = delta['tokens_changed']
    delta_bytes = archives[candidate]['bytes'] - archives['control']['bytes']
    delta_stream = len(streams[candidate]) - len(streams['control'])
    first_order = (sum(ideal[candidate]['per_frame'][1]) - sum(ideal['control']['per_frame'][1])) / 8
    break_even = inputs['break_even_bits_per_changed_token']
    bits_per_token = delta_bytes * 8 / changed if changed else None
    if seg_gain_bytes is None:
        seg_gain_bytes = 301.72664703369134
    if seg_cells is not None:
        # A SUBSET repairs fewer cells and therefore carries its own break-even; reusing the
        # full pass's 10.2715 would price the subset against a gain it does not deliver.
        break_even = seg_cells * (100.0 / (N_PAIRS * EVAL_H * EVAL_W)) / (25 / 37_545_489) * 8 / changed
    dS_rate = delta_bytes * 25 / 37_545_489
    dS_seg = -seg_gain_bytes * 25 / 37_545_489

    result = dict(
        schema='ddm_sj1_pass5_price.v1', axis=AXIS, score_claim=False, tags=list(tags),
        control_identity_passed=True, twin_sha256=twins,
        live_archive_bytes=live_bytes, archives=archives,
        stream_bytes={k: len(v) for k, v in streams.items()},
        delta_archive_bytes=delta_bytes, delta_stream_bytes=delta_stream,
        first_order_ideal_delta_bytes=first_order,
        realized_over_first_order=(delta_bytes / first_order) if first_order else None,
        candidate_field=candidate, seg_cells=seg_cells,
        tokens_changed=changed, pairs_changed=delta['pairs_changed'],
        bits_per_changed_token=bits_per_token,
        break_even_bits_per_changed_token=break_even,
        margin_ratio=break_even / bits_per_token if bits_per_token else None,
        preregistered_band_bits=inputs['preregistered_band_bits'],
        seg_gain_bytes_equivalent=seg_gain_bytes,
        dS_rate=dS_rate, dS_seg=dS_seg, dS_rate_plus_seg=dS_rate + dS_seg,
        admit_bar_net_dS=inputs['admit_bar_net_dS'],
        rate_seg_clears_bar_before_pose=bool(dS_rate + dS_seg < inputs['admit_bar_net_dS']),
    )
    return record(ROOT / f'PRICE_{candidate}.json', result)


def ledger() -> dict:
    """Per-pair ideal-bit cost of the pass-5 edits -- a RANKING for the subset sweep."""
    inputs = guard()
    per_pair_tokens = inputs['field_delta']['per_pair']
    runs = {name: load_encode(name, 'primary') for name in FIELDS}
    control = np.asarray(runs['control']['per_frame_ideal_bits'][1], dtype=np.float64)
    candidate = np.asarray(runs['candidate']['per_frame_ideal_bits'][1], dtype=np.float64)
    delta_bits = candidate - control
    rows = [dict(pair=i, tokens=int(per_pair_tokens[i]), delta_bits=float(delta_bits[i]),
                 delta_bytes=float(delta_bits[i] / 8))
            for i in range(N_PAIRS)]
    # ddm_sj1_joint_admission's `admit --bits-control/--bits-candidate` reads two (600,)
    # float vectors of per-frame bits.  Emitting them here keeps the admission on the
    # MEASURED per-pair ledger rather than the uniform bytes-per-changed-token fallback
    # that `token_rate_model_direction_dependence_v1` measured wrong by 2.24x.
    ledger_files = {}
    for name, vector in (('control', control), ('candidate', candidate)):
        destination = ROOT / 'retained' / f'bits_mixed_{name}.npy'
        preflight(destination, vector.nbytes + 4096)
        np.save(destination, vector)
        ledger_files[name] = jg2.file_fact(destination)
    return record(ROOT / 'LEDGER.json',
                  dict(schema='ddm_sj1_pass5_ledger.v1', axis=AXIS, score_claim=False,
                       note='ideal-bit deltas RANK pairs; a selected subset is priced by its own real encode',
                       total_delta_bytes=float(delta_bits.sum() / 8),
                       bit_ledgers=ledger_files, rows=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['init', 'add-field', 'encode', 'price', 'ledger'])
    parser.add_argument('--field', default='control',
                        help='a field name registered in INPUTS.json by init or add-field')
    parser.add_argument('--npz', type=Path, default=None, help='add-field: the field npz')
    parser.add_argument('--candidate', default='candidate', help='price: which field to price')
    parser.add_argument('--seg-cells', type=float, default=None,
                        help='price: cells this field repairs, so the break-even is its own')
    parser.add_argument('--seg-gain-bytes', type=float, default=None)
    parser.add_argument('--tag', default='primary')
    parser.add_argument('--stop', type=int, default=N_PAIRS)
    parser.add_argument('--tags', nargs='+', default=['primary', 'repeat'])
    args = parser.parse_args()
    if args.stage == 'init':
        print(json.dumps(initialize(), sort_keys=True)[:2000])
    elif args.stage == 'add-field':
        if args.npz is None:
            raise SystemExit('add-field requires --npz')
        out = add_field(args.field, args.npz)
        print(json.dumps({k: v['u8'] for k, v in out['fields'].items()}, sort_keys=True))
    elif args.stage == 'encode':
        out = encode(args.field, args.tag, args.stop)
        print(json.dumps({k: v for k, v in out.items() if k != 'per_frame_ideal_bits'}, sort_keys=True))
    elif args.stage == 'price':
        print(json.dumps(price(tuple(args.tags), args.candidate,
                               args.seg_gain_bytes, args.seg_cells), sort_keys=True))
    else:
        out = ledger()
        print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, sort_keys=True))
