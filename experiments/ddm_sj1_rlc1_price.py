#!/usr/bin/env python3
"""Price an EDITED token field by REAL re-encode under the RLC1 coder the pointer ships.

WHY THIS EXISTS
---------------
``experiments/ddm_sj1_pass5_price.py`` prices through ``ddm_tc1_mixer_codec.SharedMixer``
over the FreeCorrector/HPAC rows.  That was the shipped coder for pointer moves 36-43.  It
is NOT the shipped coder for move 48, and the difference is not cosmetic:

* move 44 (rlc5) replaced the raw ``TC1M`` rider with a counted ``RLC1`` one;
* move 45 refit the CAP1 predictor;
* move 47 replaced the HPAC prior with the retrained one;
* moves 46/48 added the frame-embedding even rounding INSIDE the receiver.

MEASURED on move 48's own field, twice, with byte-identical twins: the tc1 route emits a
118,929 B stream where the archive ships 118,896 B -- **+33 B, and a different sha**.  The
pass-5 pricer's own gate 2 refused it ("CONTROL IDENTITY FAILED"), which is the gate doing
its job: a byte delta measured on a coder the archive does not use is not this candidate's
price ([[first_order_token_price_is_a_ranking_never_a_charge...]] is about first-order sums;
this is the stronger sibling -- the wrong CODER, not the wrong ESTIMATOR).

WHAT THIS DOES INSTEAD
----------------------
It drives the RECEIVER'S OWN decode loop -- ``rx.decode_production_tokens`` on the pointer
tree's own runtime -- with the true symbols injected at ``NativeDecoder.decode``, and feeds
those symbols and the receiver's own probability rows to twin arithmetic encoders.  The
coder is therefore not re-implemented here at all: it is the shipped one, loaded from the
shipped tree, and this module only supplies the symbols and collects the output.  That is
``experiments/ddm_hpr1_shape_price.py``'s mechanism verbatim; the one thing this module adds
is that the FIELD is a parameter, because hpr1's rail prices a MODEL change on a fixed field
and this arm prices a FIELD change on a fixed model.  Editing hpr1's rail to parameterise it
was rejected on purpose: it binds its own ``producer`` sha into every INPUTS.json and every
encoder checkpoint receipt, so an edit would refuse every in-flight dpi1/hpr1 resume
([[binding_hash_whole_module_kills_checkpoints_20260909]]).

THE GATE THAT MAKES THIS HARNESS CREDIBLE
-----------------------------------------
``encode --field control`` re-encodes the pointer's OWN field and the packed archive must
come back byte-identical to the pointer's archive (sha AND size).  If this harness were not
equivalent to the shipped loop, that control could not pass.  It is checked here, in code,
and no candidate price is returned until it has passed in this store.

No scorer, no dispatch, no pointer write, no edit to any sister arm's tree.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'experiments')]
sys.dont_write_bytecode = True
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np  # noqa: E402

from experiments import ddm_jg2_tail_reencode as jg2  # noqa: E402
from experiments.ddm_tc1_public_proof import build_libraries  # noqa: E402

#: ROOT/BULK/LIVE are env-overridable rather than edited, for the reason the docstring
#: gives: this module binds its own sha into every checkpoint receipt, so a source edit
#: while an encode is in flight kills that encode's resume.  A generation is a new ROOT
#: plus the new pointer's tree; INPUTS.json then pins both by sha.
ROOT = Path(os.environ.get('SJ1_RLC1_ROOT', '/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/rlc1_price'))
BULK = Path(os.environ.get('SJ1_RLC1_BULK', '/Volumes/APDataStore/pact/ddm_sj1_pass7/rlc1'))
LIVE = Path(os.environ.get(
    'SJ1_RLC1_LIVE',
    '/Volumes/VertigoDataTier/pact/ddm_hpr1/public/retrain_frame_even/candidate_runtime'))
CONTROL_NPZ = Path(os.environ.get(
    'SJ1_RLC1_CONTROL_NPZ',
    '/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/retained/field_move48.npz'))
POINTER = REPO / '.omx/state/canonical_frontier_pointer.json'
AXIS = '[macOS-CPU advisory / scorer-free EXACT byte measurement]'
TAIL_PREFIX_BYTES = 96
RIDER_MAGIC = b'RLC1'
N_PAIRS, EVAL_H, EVAL_W = 600, 384, 512
#: Never let a price run take the volume below this; the pass's payloads live here too.
RESERVE_BYTES = 8 << 30


class Rlc1PriceError(RuntimeError):
    """A pricing input or invariant is not what the shipped object says it is."""


# ----------------------------------------------------------------------------------
# custody helpers -- same contract as ddm_sj1_pass5_price
# ----------------------------------------------------------------------------------

def preflight(path: Path, need: int = 1 << 20) -> None:
    """Refuse a write outside the owned stores, or one the volume cannot hold."""
    path = Path(path).resolve()
    if not (path.is_relative_to(ROOT) or path.is_relative_to(BULK)):
        raise Rlc1PriceError(f'write outside owned stores: {path}')
    mount = Path('/Volumes') / path.parts[2]
    if shutil.disk_usage(mount).free < need + RESERVE_BYTES:
        raise Rlc1PriceError(f'storage preflight refused on {mount}: need {need} B over an 8 GiB reserve')
    path.parent.mkdir(parents=True, exist_ok=True)


def write(path: Path, payload: bytes) -> dict:
    path = Path(path)
    if path.exists():
        if path.read_bytes() != payload:
            raise Rlc1PriceError(f'immutable payload changed: {path}')
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
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc' \
                and not path.name.startswith('._'):
            facts[str(path.relative_to(root))] = jg2.file_fact(path)
    return facts


def field_to_u8(npz_path: Path, destination: Path) -> dict:
    with np.load(jg2.resolve_artifact(npz_path), allow_pickle=False) as data:
        if set(data.files) != {str(i) for i in range(N_PAIRS)}:
            raise Rlc1PriceError(f'{npz_path} does not carry all {N_PAIRS} planes')
        field = np.stack([data[str(i)] for i in range(N_PAIRS)])
    if field.shape != (N_PAIRS, EVAL_H, EVAL_W) or field.dtype != np.uint8:
        raise Rlc1PriceError(f'{npz_path} geometry is wrong: {field.shape} {field.dtype}')
    return write(destination, field.tobytes())


# ----------------------------------------------------------------------------------
# the runtime copy -- the pointer tree is READ-ONLY, so the loop runs on a pinned copy
# ----------------------------------------------------------------------------------

def runtime_copy() -> Path:
    """Copy the live tree into this store once, and prove the copy is the live archive."""
    destination = ROOT / 'runtime_copy'
    live_sha = jg2.sha256_file(LIVE / 'archive.zip')
    for src in sorted(LIVE.rglob('*')):
        if not src.is_file() or '__pycache__' in src.parts or src.suffix == '.pyc' \
                or src.name.startswith('._'):
            continue
        write(destination / src.relative_to(LIVE), src.read_bytes())
    if jg2.sha256_file(destination / 'archive.zip') != live_sha:
        raise Rlc1PriceError('the copied archive is not the live archive')
    return destination


def build_geometry(runtime: Path, work: Path) -> dict:
    """Compile the RLC1 geometry library exactly as ``inflate.sh`` does."""
    target = work / 'rlc1_geometry.so'
    source = runtime / 'runtime/rlc1_geometry.c'
    command = [os.environ.get('CC', 'cc'), '-O3', '-std=c11', '-shared', '-fPIC',
               str(source), '-o', str(target)]
    work.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (work / 'rlc1_geometry.build.log').write_text(result.stdout + result.stderr)
        result.check_returncode()
    os.environ['RLC1_GEOMETRY_LIBRARY'] = str(target)
    return {'source': jg2.file_fact(source), 'library': jg2.file_fact(target), 'argv': command}


# ----------------------------------------------------------------------------------
# init / add-field
# ----------------------------------------------------------------------------------

def split_tail(tail: bytes, stream_bytes: int) -> tuple[bytes, bytes, bytes]:
    """Return (prefix, RIDER, stream); the rider travels VERBATIM, never rebuilt."""
    prefix = tail[:TAIL_PREFIX_BYTES]
    body = tail[TAIL_PREFIX_BYTES:]
    rider, stream = body[:len(body) - stream_bytes], body[len(body) - stream_bytes:]
    if rider[:len(RIDER_MAGIC)] != RIDER_MAGIC:
        raise Rlc1PriceError(f'the shipped rider magic is {rider[:4]!r}, not {RIDER_MAGIC!r}')
    return prefix, rider, stream


def guard() -> dict:
    inputs = json.loads((ROOT / 'INPUTS.json').read_text())
    live = json.loads(POINTER.read_text())['our_local_frontier_contest_cuda']
    if live['archive_sha256'] != inputs['pointer_archive']['sha256']:
        raise Rlc1PriceError('POINTER_MOVED: refuse; the price must be measured against the live row')
    for relative, fact in inputs['runtime_sources'].items():
        if jg2.file_fact(LIVE / relative) != fact:
            raise Rlc1PriceError(f'live runtime drift at {relative}')
    for name, entry in inputs['fields'].items():
        if jg2.file_fact(Path(entry['u8']['path'])) != entry['u8']:
            raise Rlc1PriceError(f'{name} field custody changed')
    return inputs


def initialize() -> dict:
    if (ROOT / 'INPUTS.json').exists():
        return guard()
    pointer = json.loads(POINTER.read_text())
    live = pointer['our_local_frontier_contest_cuda']
    archive = jg2.file_fact(LIVE / 'archive.zip')
    if archive['sha256'] != live['archive_sha256']:
        raise Rlc1PriceError('configured tree is not the live pointer')

    copy = runtime_copy()
    residual, _renderer, _code_dir = jg2.load_runtime(copy)
    parts = residual.read_residual_archive(copy / 'archive.zip')
    stream = bytes(parts.token_stream)
    member = jg2.read_archive_member(copy / 'archive.zip')
    sections = jg2.split_member(member)
    prefix, rider, split_stream = split_tail(sections['tail'], len(stream))
    if split_stream != stream:
        raise Rlc1PriceError('the tail suffix is not the stream the receiver returns')
    if prefix + rider + stream != sections['tail']:
        raise Rlc1PriceError('tail split is not lossless')
    if rider != RIDER_MAGIC + bytes(parts.tc1_weights):
        raise Rlc1PriceError('the rider is not magic+weights; the encode path would rebuild it wrong')

    null = ROOT / 'retained/null_archive.zip'
    preflight(null, archive['bytes'])
    jg2.pack_archive(member, null)
    if null.read_bytes() != (LIVE / 'archive.zip').read_bytes():
        raise Rlc1PriceError('null rebuild differs from the live archive')

    census = {k: write(ROOT / 'retained/source' / (k + '.bin'), v) for k, v in sections.items()}
    fields = {'control': dict(npz=jg2.file_fact(CONTROL_NPZ),
                              u8=field_to_u8(CONTROL_NPZ, BULK / 'fields/control.u8'))}
    result = dict(
        schema='ddm_sj1_rlc1_price_inputs.v1', axis=AXIS, score_claim=False, seed=20260912,
        pointer=pointer, pointer_archive=archive, live_tree=str(LIVE), runtime_copy=str(copy),
        runtime_sources=runtime_facts(LIVE), census=census,
        tail=dict(bytes=len(sections['tail']), prefix_bytes=len(prefix),
                  rider_bytes=len(rider), rider_sha256=jg2.sha256_bytes(rider),
                  rider_magic=RIDER_MAGIC.decode(),
                  stream_bytes=len(stream), stream_sha256=jg2.sha256_bytes(stream)),
        fields=fields,
        admit_bar_net_dS=-2e-05,
        storage_policy='Vertigo receipts + streams + archives; APDataStore fields and encoder states',
    )
    return record(ROOT / 'INPUTS.json', result)


def add_field(name: str, npz_path: Path) -> dict:
    """Register a further field for pricing -- the pass field, or a Lagrange subset."""
    inputs = guard()
    if name in inputs['fields']:
        raise Rlc1PriceError(f'field {name} is already registered; fields are immutable here')
    if not name.isidentifier():
        raise Rlc1PriceError('field name must be a plain identifier')
    u8 = field_to_u8(npz_path, BULK / 'fields' / f'{name}.u8')
    control = np.fromfile(inputs['fields']['control']['u8']['path'],
                          dtype=np.uint8).reshape(N_PAIRS, -1)
    plane = np.fromfile(u8['path'], dtype=np.uint8).reshape(N_PAIRS, -1)
    per_pair = (control != plane).sum(axis=1)
    inputs['fields'][name] = dict(
        npz=jg2.file_fact(npz_path), u8=u8,
        delta_vs_control=dict(tokens_changed=int(per_pair.sum()),
                              pairs_changed=int((per_pair > 0).sum()),
                              per_pair=per_pair.astype(int).tolist()))
    return record(ROOT / 'INPUTS.json', inputs)


# ----------------------------------------------------------------------------------
# encode -- the SHIPPED RLC1 loop, symbols injected, twin arithmetic encoders
# ----------------------------------------------------------------------------------

def encode(field_name: str, tag: str) -> dict:
    import torch

    inputs = guard()
    if field_name not in inputs['fields']:
        raise Rlc1PriceError(f'field {field_name} is not registered')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(inputs['seed'])
    np.random.seed(inputs['seed'])
    torch.use_deterministic_algorithms(True)

    work = ROOT / 'encode' / field_name / tag
    preflight(work / 'build')
    runtime = Path(inputs['runtime_copy'])
    route = jg2.load_route_b()
    build_path = work / 'ENCODER_BUILD.json'
    if build_path.exists():
        build = json.loads(build_path.read_text())
        library = Path(build['library']['path'])
    else:
        library, build = jg2.compile_rc64(work, route, f'sj1rlc1_{field_name}_{tag}')
        record(build_path, build)
    build_libraries(runtime, work / 'native')
    geometry = build_geometry(runtime, work / 'native')

    rx, renderer, code_dir = jg2.load_runtime(runtime)
    parts = rx.read_residual_archive(runtime / 'archive.zip')
    if jg2.sha256_bytes(bytes(parts.token_stream)) != inputs['tail']['stream_sha256']:
        raise RuntimeError("the live tree's own reader no longer returns the pinned tail stream")

    checkpoints = work / 'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    os.environ['TC1_RECEIVER_CHECKPOINT_DIR'] = str(checkpoints)
    os.environ['TC1_RECEIVER_STOP_AFTER'] = str(N_PAIRS)
    from runtime.entropy.rc64 import NativeDecoder
    from runtime.rlc1_mixer import LaneMixer
    from runtime.tc1_receiver_checkpoint import ReceiverCheckpoint

    binding = dict(inputs_sha=jg2.sha256_file(ROOT / 'INPUTS.json'),
                   source_sha=jg2.sha256_file(Path(__file__)),
                   jg2_sha=jg2.sha256_file(Path(jg2.__file__)),
                   field=inputs['fields'][field_name]['u8']['sha256'],
                   runtime_sources={k: v['sha256'] for k, v in inputs['runtime_sources'].items()},
                   build=build)

    field = np.memmap(inputs['fields'][field_name]['u8']['path'], dtype=np.uint8,
                      mode='r', shape=(N_PAIRS, EVAL_H, EVAL_W))
    per_frame_bits = np.zeros(N_PAIRS, dtype=np.float64)
    state, start = None, 0
    latest = checkpoints / 'LATEST.json'
    if latest.exists():
        start = int(json.loads(latest.read_text())['frame'])
        state_path = BULK / 'encoder_states' / field_name / tag / f'stage_{start:04d}.npz'
        receipt = json.loads(state_path.with_suffix('.json').read_text())
        if jg2.file_fact(state_path) != receipt['payload'] or receipt['binding'] != binding:
            raise Rlc1PriceError('encoder restart binding mismatch')
        with np.load(state_path, allow_pickle=False) as data:
            state = {k: data[k] for k in data.files}
        per_frame_bits = state['per_frame_bits'].copy()

    twins = [route.NativeRc64Encoder(library, None if state is None else state[f'enc{i}'].tobytes())
             for i in range(2)]
    observed = {'frame': start, 'positions': None}
    started = time.monotonic()
    original_coding, original_end = LaneMixer.coding, LaneMixer.end_frame
    original_save = ReceiverCheckpoint.save

    def coding(self, rows, positions, plane, previous):
        if observed['positions'] is not None or self.frame != observed['frame']:
            raise Rlc1PriceError('known-symbol group order mismatch')
        observed['positions'] = positions.copy()
        return original_coding(self, rows, positions, plane, previous)

    def known_symbols(self, probabilities):
        positions = observed['positions']
        if positions is None or len(positions) != len(probabilities):
            raise Rlc1PriceError('known-symbol decode call lacks its group')
        symbols = field[observed['frame']].reshape(-1)[positions].astype(np.int32)
        rows = np.asarray(probabilities, dtype=np.float64)
        coded = np.clip(rows[np.arange(len(symbols)), symbols], 1e-12, 1.0)
        per_frame_bits[observed['frame']] += float(-np.log2(coded).sum())
        for encoder in twins:
            encoder.encode(symbols, probabilities)
        observed['positions'] = None
        return symbols

    def end(self, plane, previous):
        # OUTPUT-LOSSLESS: the plane the receiver reconstructs is the field we are pricing,
        # frame by frame.  A field the receiver cannot reproduce has no price.
        np.testing.assert_array_equal(plane, field[observed['frame']])
        original_end(self, plane, previous)
        observed['frame'] += 1

    def save(self, frame, tokens):
        if frame != observed['frame'] or observed['positions'] is not None:
            raise Rlc1PriceError('encoder checkpoint is not at a frame boundary')
        path = BULK / 'encoder_states' / field_name / tag / f'stage_{frame:04d}.npz'
        values = {f'enc{i}': np.frombuffer(e.snapshot(), dtype=np.uint8) for i, e in enumerate(twins)}
        values['per_frame_bits'] = per_frame_bits.copy()
        preflight(path, sum(v.nbytes for v in values.values()) + (1 << 20))
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                if set(old.files) != set(values) or any(
                        not np.array_equal(old[k], v) for k, v in values.items()):
                    raise Rlc1PriceError('immutable checkpoint differs')
        else:
            temporary = path.with_suffix('.npz.new')
            with temporary.open('wb') as handle:
                np.savez(handle, **values)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        record(path.with_suffix('.json'), {'payload': jg2.file_fact(path),
                                           'binding': binding, 'frame': frame})
        (checkpoints / 'LATEST.json.new').write_text(json.dumps({'frame': frame}))
        os.replace(checkpoints / 'LATEST.json.new', latest)
        print(json.dumps(dict(field=field_name, tag=tag, frame=frame,
                              seconds=round(time.monotonic() - started, 1))), flush=True)
        original_save(self, frame, tokens)

    LaneMixer.coding, LaneMixer.end_frame = coding, end
    NativeDecoder.decode, ReceiverCheckpoint.save = known_symbols, save
    try:
        tokens, _ = rx.decode_production_tokens(parts, renderer, code_dir, torch.device('cpu'))
    finally:
        LaneMixer.coding, LaneMixer.end_frame = original_coding, original_end
        ReceiverCheckpoint.save = original_save
    decoded_sha = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    if observed['frame'] != N_PAIRS or decoded_sha != inputs['fields'][field_name]['u8']['sha256']:
        raise Rlc1PriceError('the known-symbol loop did not reproduce the priced field over 600 frames')

    outputs, archives = {}, []
    sections = jg2.split_member(jg2.read_archive_member(runtime / 'archive.zip'))
    for index, encoder in enumerate(twins):
        outputs[f'envelope{index}'] = write(work / f'envelope_{index}.bin', encoder.finish())
        raw = ctypes.string_at(encoder.library.rc64_encoder_data(encoder.context),
                               int(encoder.library.rc64_encoder_size(encoder.context)))
        outputs[f'stream{index}'] = write(work / f'stream_{index}.rc64', raw)
        rider = RIDER_MAGIC + bytes(parts.tc1_weights)
        changed = dict(sections)
        changed['tail'] = sections['tail'][:TAIL_PREFIX_BYTES] + rider + raw
        member = jg2.join_member(changed)
        write(work / f'member_{index}.bin', member)
        archive = work / f'archive_{index}.zip'
        if not archive.exists():
            preflight(archive, len(member) + 4096)
            jg2.pack_archive(member, archive)
        parsed = rx.read_residual_archive(archive)
        if bytes(parsed.token_stream) != raw:
            raise Rlc1PriceError('the packed archive does not parse back to the emitted stream')
        for name in ('hpac_blob', 'semantic_blob', 'carrier_blob', 'tc1_weights', 'residual_payload'):
            if getattr(parts, name) != getattr(parsed, name):
                raise Rlc1PriceError(f'unrelated archive component changed: {name}')
        archives.append(jg2.file_fact(archive))
    if archives[0]['sha256'] != archives[1]['sha256']:
        raise Rlc1PriceError('TWIN FAILED: the two in-process encoders disagree')

    identity = archives[0]['sha256'] == inputs['pointer_archive']['sha256']
    result = dict(schema='ddm_sj1_rlc1_encode.v1', axis=AXIS, score_claim=False,
                  field=field_name, tag=tag, frames=N_PAIRS, start_frame=start,
                  geometry_build=geometry, outputs=outputs, archives=archives,
                  stream_bytes=archives[0]['bytes'] and len(
                      (work / 'stream_0.rc64').read_bytes()),
                  decoded_field_sha256=decoded_sha, output_lossless=True,
                  archive_matches_live_pointer=identity,
                  per_frame_bits=per_frame_bits.tolist(),
                  ideal_bytes=float(per_frame_bits.sum() / 8.0), binding=binding)
    record(work / 'ENCODE.json', result)
    if field_name == 'control' and not identity:
        raise Rlc1PriceError(
            'CONTROL IDENTITY FAILED: re-encoding the pointer\'s own field through this '
            f'harness gives {archives[0]["sha256"]} ({archives[0]["bytes"]} B), not the '
            f'pointer\'s {inputs["pointer_archive"]["sha256"]} '
            f'({inputs["pointer_archive"]["bytes"]} B)')
    return result


# ----------------------------------------------------------------------------------
# price -- the EXACT container delta, and the per-pair bit ledger the admission reads
# ----------------------------------------------------------------------------------

def load_encode(field_name: str, tag: str) -> dict:
    path = ROOT / 'encode' / field_name / tag / 'ENCODE.json'
    if not path.exists():
        raise Rlc1PriceError(f'missing completed encode: {path}')
    return json.loads(path.read_text())


def price(tags: tuple[str, ...], candidate: str, seg_cells: float | None = None) -> dict:
    inputs = guard()
    control_runs = [load_encode('control', tag) for tag in tags]
    if not all(run['archive_matches_live_pointer'] for run in control_runs):
        raise Rlc1PriceError('CONTROL IDENTITY FAILED at price time')
    runs = {'control': control_runs, candidate: [load_encode(candidate, tag) for tag in tags]}
    archives, twins, ideal = {}, {}, {}
    for name, group in runs.items():
        shas = {run['archives'][0]['sha256'] for run in group}
        if len(shas) != 1:
            raise Rlc1PriceError(f'{name}: independent encode PROCESSES disagree; the delta would be variance')
        archives[name] = group[0]['archives'][0]
        twins[name] = sorted(shas) * len(group)
        ideal[name] = group[0]

    live_bytes = inputs['pointer_archive']['bytes']
    if archives['control']['bytes'] != live_bytes:
        raise Rlc1PriceError('rebuilt control archive does not reproduce the live byte count')
    entry = inputs['fields'][candidate]
    delta = entry['delta_vs_control']
    changed = delta['tokens_changed']
    delta_bytes = archives[candidate]['bytes'] - archives['control']['bytes']
    first_order = (ideal[candidate]['ideal_bytes'] - ideal['control']['ideal_bytes'])
    bits_per_token = delta_bytes * 8 / changed if changed else None
    break_even = None
    if seg_cells is not None:
        # A subset repairs fewer cells and therefore carries its OWN break-even.
        break_even = seg_cells * (100.0 / (N_PAIRS * EVAL_H * EVAL_W)) / (25 / 37_545_489) * 8 / changed
    dS_rate = delta_bytes * 25 / 37_545_489
    result = dict(
        schema='ddm_sj1_rlc1_price.v1', axis=AXIS, score_claim=False, tags=list(tags),
        control_identity_passed=True, twin_sha256=twins,
        live_archive_bytes=live_bytes, archives=archives,
        candidate_field=candidate, seg_cells=seg_cells,
        tokens_changed=changed, pairs_changed=delta['pairs_changed'],
        delta_archive_bytes=delta_bytes,
        first_order_ideal_delta_bytes=first_order,
        realized_over_first_order=(delta_bytes / first_order) if first_order else None,
        bits_per_changed_token=bits_per_token,
        break_even_bits_per_changed_token=break_even,
        margin_ratio=(break_even / bits_per_token) if (break_even and bits_per_token) else None,
        dS_rate=dS_rate, admit_bar_net_dS=inputs['admit_bar_net_dS'],
    )
    return record(ROOT / f'PRICE_{candidate}.json', result)


def ledger(candidate_field: str) -> dict:
    """Per-pair ideal-bit cost of a field's edits -- the RANKING the subset sweep reads."""
    inputs = guard()
    control = np.asarray(load_encode('control', 'primary')['per_frame_bits'], dtype=np.float64)
    candidate = np.asarray(load_encode(candidate_field, 'primary')['per_frame_bits'], dtype=np.float64)
    files = {}
    for name, vector in (('control', control), (candidate_field, candidate)):
        destination = ROOT / 'retained' / f'bits_rlc1_{name}.npy'
        preflight(destination, vector.nbytes + 4096)
        np.save(destination, vector)
        files[name] = jg2.file_fact(destination)
    per_pair_tokens = inputs['fields'][candidate_field]['delta_vs_control']['per_pair']
    rows = [dict(pair=i, tokens=int(per_pair_tokens[i]),
                 delta_bits=float(candidate[i] - control[i]),
                 delta_bytes=float((candidate[i] - control[i]) / 8))
            for i in range(N_PAIRS)]
    return record(ROOT / 'LEDGER.json',
                  dict(schema='ddm_sj1_rlc1_ledger.v1', axis=AXIS, score_claim=False,
                       note='ideal-bit deltas RANK pairs; a selected subset is priced by its own real encode',
                       total_delta_bytes=float((candidate - control).sum() / 8),
                       bit_ledgers=files, rows=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['init', 'add-field', 'encode', 'price', 'ledger'])
    parser.add_argument('--field', default='control', help='encode: which registered field')
    parser.add_argument('--npz', type=Path, default=None, help='add-field: the field npz')
    parser.add_argument('--name', default=None, help='add-field: the name to register it under')
    parser.add_argument('--candidate', default=None, help='price/ledger: which field to price')
    parser.add_argument('--seg-cells', type=float, default=None,
                        help='price: the subset\'s own repaired-cell count, for its own break-even')
    parser.add_argument('--tag', default='primary')
    parser.add_argument('--tags', nargs='+', default=['primary', 'repeat'])
    args = parser.parse_args()
    if args.stage == 'init':
        print(json.dumps(initialize(), sort_keys=True)[:2000])
    elif args.stage == 'add-field':
        if args.npz is None or args.name is None:
            raise SystemExit('add-field needs --name and --npz')
        add_field(args.name, args.npz)
        print(json.dumps({'registered': args.name}))
    elif args.stage == 'encode':
        print(json.dumps({k: v for k, v in encode(args.field, args.tag).items()
                          if k != 'per_frame_bits'}, sort_keys=True)[:2000])
    elif args.stage == 'price':
        if args.candidate is None:
            raise SystemExit('price needs --candidate')
        print(json.dumps(price(tuple(args.tags), args.candidate, args.seg_cells), sort_keys=True)[:2000])
    else:
        if args.candidate is None:
            raise SystemExit('ledger needs --candidate')
        print(json.dumps({k: v for k, v in ledger(args.candidate).items() if k != 'rows'},
                         sort_keys=True))
