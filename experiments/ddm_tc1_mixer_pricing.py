#!/usr/bin/env python3
"""Price the one admitted TC1 mixer on the complete retained source field."""
from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments import ddm_jg2_tail_reencode as jg2
from experiments import ddm_tc1_mixer_codec as codec
from experiments.ddm_tc1_shared_predictor_bound import KEEP_MISS, PREDICTOR_STORE, objective_function
from experiments.ddm_tc1_token_tail_bound import AXIS, ROOT, pinned


def admitted(root):
    inputs = pinned(root)
    bound = json.loads((root / PREDICTOR_STORE / 'BOUND_FINAL.json').read_text())
    if not bound['full_n600'] or bound['upper_net_ideal_bytes'] < 500:
        raise RuntimeError('TC1 construction gate refuses this bound')
    if not json.loads((root / 'TRACE_0600.json').read_text())['full_control_byte_identical']:
        raise RuntimeError('source trace did not reproduce the shipped stream')
    return inputs, bound


def quantize(root):
    """One deterministic int8 coordinate pass; every trial and weight is retained."""
    inputs, bound = admitted(root)
    store = root / 'mixer/quantize'
    store.mkdir(parents=True, exist_ok=True)
    objective, arrays, build = objective_function(root, codec.N)
    weights = np.clip(np.rint(np.array(bound['weights']) * codec.SCALE), -128, 127).astype(np.int8)
    jg2.persist_immutable_bytes(store / 'rounded_initial_i8.bin', weights.tobytes(), label='initial counted weight quantization')
    state_path = store / 'STATE.json'
    start = 0
    if state_path.exists():
        state = json.loads(state_path.read_text())
        start = state['next_coordinate']; weights = np.array(state['weights'], dtype=np.int8)
    initial = objective(weights.astype(float) / codec.SCALE, False)[0]
    loss = initial
    for coordinate in range(start, codec.K * codec.F):
        bank, feature = divmod(coordinate, codec.F)
        trials = []
        best = weights.copy(); best_loss = loss
        for delta in (-1, 1):
            value = int(weights[bank, feature]) + delta
            if not -128 <= value <= 127:
                continue
            candidate = weights.copy(); candidate[bank, feature] = value
            candidate_loss = objective(candidate.astype(float) / codec.SCALE, False)[0]
            trials.append(dict(weights=candidate.tolist(), loss_nats=candidate_loss))
            if candidate_loss < best_loss - 1e-9:
                best, best_loss = candidate, candidate_loss
        jg2.atomic_json(store / f'TRIALS_{coordinate:02d}.json', trials)
        weights, loss = best, best_loss
        jg2.atomic_json(state_path, dict(next_coordinate=coordinate + 1, weights=weights.tolist(), loss_nats=loss))
        if (coordinate + 1) % 7 == 0:
            print(json.dumps(dict(stage='int8_coordinate_pass', coordinates=coordinate + 1, loss_nats=loss)), flush=True)
    payload = weights.tobytes()
    target = root / 'mixer/weights_i8.bin'
    jg2.persist_immutable_bytes(target, payload, label='35 counted TC1 weights')
    result = dict(axis=AXIS, score_claim=False, coefficient_count=35,
                  weights=jg2.file_fact(target), weight_values=weights.tolist(),
                  kept_loss_nats=loss, omitted_positions_not_used_for_fitting=True,
                  final_pricing='all 117964800 positions through RC64, including omitted-fit positions',
                  build=build, archive=inputs['archive'])
    jg2.atomic_json(root / 'mixer/QUANTIZATION.json', result)
    return result


def verify_codec(root):
    """Check the portable reader's grouped causal path against retained real rows."""
    inputs, bound = admitted(root)
    store = root / 'mixer/verification'; store.mkdir(parents=True, exist_ok=True)
    weight_bytes = np.rint(np.array(bound['weights']) * codec.SCALE).astype(np.int8).tobytes()
    jg2.persist_immutable_bytes(store / 'weights.bin', weight_bytes, label='codec validation weights')
    mixer = codec.SharedMixer(weight_bytes)
    tokens = jg2.load_tokens(Path(inputs['field']['path']))
    order = np.argsort(codec.GROUP, kind='stable')
    route = jg2.load_route_b()
    library, build = jg2.compile_rc64(store, route, 'tc1_verify')
    encoded = route.NativeRc64Encoder(library)
    coding_planes = []
    for frame in range(2):
        with np.load(root / 'rows' / f'frame_{frame:04d}.npz', allow_pickle=False) as data:
            rows = data['rows']
        plane, previous = tokens[frame], None if not frame else tokens[frame - 1]
        mixer.begin_frame()
        phi, freq = mixer.features(rows, codec.ALL, plane, previous)
        truth = plane.reshape(-1)
        keep = ((1 - freq.max(axis=1).astype(float) / codec.TOTAL) >= KEEP_MISS) | (truth != freq.argmax(axis=1))
        with np.load(root / PREDICTOR_STORE / 'frames' / f'frame_{frame:04d}.npz', allow_pickle=False) as reference:
            np.testing.assert_array_equal(phi[keep], reference['phi'])
        full = codec.mix_probabilities(freq, phi, mixer.weights, rows)
        zero = codec.mix_probabilities(freq, phi, np.zeros_like(mixer.weights), rows)
        np.testing.assert_array_equal(codec.frequencies(zero), freq)
        partial = np.zeros_like(plane).reshape(-1)
        seen = np.zeros(codec.H * codec.W, dtype=bool)
        for group in np.unique(codec.GROUP):
            positions = np.flatnonzero(group == codec.GROUP)
            observed = mixer.coding(rows[positions], positions, partial, previous)
            np.testing.assert_array_equal(observed, full[positions])
            partial[positions] = truth[positions]
            seen[positions] = True
        if not seen.all():
            raise RuntimeError('group plan omitted symbols')
        jg2.atomic_npz(store / f'frame_{frame:04d}.npz', dict(mixed=full, zero=zero))
        coding_planes.append(full)
        encoded.encode(truth[order].astype(np.int32), full[order])
        mixer.end_frame(plane, previous)
    envelope = encoded.finish()
    jg2.persist_immutable_bytes(store / 'two_frame_control.rc64', envelope, label='codec differential test stream')
    decoder = route.NativeRc64Decoder(library, envelope)
    decoded = []
    for frame, coding in enumerate(coding_planes):
        symbols = decoder.decode(None, coding[order])
        plane = np.empty(codec.H * codec.W, dtype=np.uint8)
        plane[order] = symbols
        decoded.append(plane.reshape(codec.H, codec.W))
    decoded = np.stack(decoded)
    jg2.atomic_npz(store / 'decoded.npz', dict(tokens=decoded))
    np.testing.assert_array_equal(decoded, tokens[:2])
    result = dict(axis=AXIS, score_claim=False, verdict='PASS',
                  scope='codec implementation differential test, not a prefix research verdict',
                  tested_real_symbols=2 * codec.H * codec.W, all190_groups_per_frame=True,
                  zero_weights_preserve_every_frequency=True, grouped_and_plane_coding_byte_identical=True,
                  receiver_decode_exact=True, build=build,
                  payloads=[jg2.file_fact(p) for p in sorted(store.glob('*.npz'))],
                  stream=jg2.file_fact(store / 'two_frame_control.rc64'))
    jg2.atomic_json(store / 'VERIFY.json', result)
    return result


def encode(root, tag):
    inputs, bound = admitted(root)
    store = root / 'mixer' / tag
    store.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 1024**3:
        raise RuntimeError('TC1 encoding requires 1 GiB of available SSD space')
    route = jg2.load_route_b()
    library, build = jg2.compile_rc64(store, route, 'tc1_' + tag)
    weight_path = root / 'mixer/weights_i8.bin'
    mixer = codec.SharedMixer(weight_path.read_bytes())
    tokens = jg2.load_tokens(Path(inputs['field']['path']))
    order = np.argsort(codec.GROUP, kind='stable')
    per_frame = np.zeros(codec.N)
    snapshots = sorted(store.glob('state_*.npz'))
    if snapshots:
        with np.load(snapshots[-1], allow_pickle=False) as data:
            state = {k: data[k] for k in data.files}
        mixer.restore(state)
        encoder = route.NativeRc64Encoder(library, state['encoder'].tobytes())
        per_frame = state['per_frame']
    else:
        encoder = route.NativeRc64Encoder(library)
    start = mixer.frame
    started = time.monotonic()
    for frame in range(start, codec.N):
        path = root / 'rows' / f'frame_{frame:04d}.npz'
        if jg2.file_fact(path) != json.loads(path.with_suffix('.json').read_text()):
            raise RuntimeError('source row payload checksum changed')
        with np.load(path, allow_pickle=False) as data:
            rows = data['rows']
        plane = tokens[frame]
        previous = None if not frame else tokens[frame - 1]
        mixer.begin_frame()
        phi, freq = mixer.features(rows, codec.ALL, plane, previous)
        truth = plane.reshape(-1)
        keep = ((1 - freq.max(axis=1).astype(float) / codec.TOTAL) >= KEEP_MISS) | (truth != freq.argmax(axis=1))
        with np.load(root / PREDICTOR_STORE / 'frames' / f'frame_{frame:04d}.npz', allow_pickle=False) as reference:
            if not np.array_equal(phi[keep], reference['phi']):
                raise RuntimeError(f'full-field receiver feature parity failed at frame {frame}')
        coding = codec.mix_probabilities(freq, phi, mixer.weights, rows)
        coded_freq = codec.frequencies(coding)
        per_frame[frame] = -np.log2(coded_freq[np.arange(len(truth)), truth].astype(float) / codec.TOTAL).sum()
        encoder.encode(truth[order].astype(np.int32), coding[order])
        mixer.end_frame(plane, previous)
        if mixer.frame % 25 == 0 or mixer.frame == codec.N:
            state = mixer.snapshot()
            state['encoder'] = np.frombuffer(encoder.snapshot(), dtype=np.uint8)
            state['per_frame'] = per_frame.copy()
            checkpoint = store / f'state_{mixer.frame:04d}.npz'
            jg2.atomic_npz(checkpoint, state)
            jg2.atomic_json(checkpoint.with_suffix('.json'), jg2.file_fact(checkpoint))
            print(json.dumps(dict(stage=tag, pairs=mixer.frame, ideal_bytes=float(per_frame.sum() / 8), elapsed_s=time.monotonic() - started)), flush=True)
    envelope = encoder.finish()
    jg2.persist_immutable_bytes(store / 'rc64_envelope.bin', envelope, label='retained RC64 envelope')
    size = int(encoder.library.rc64_encoder_size(encoder.context))
    raw = ctypes.string_at(encoder.library.rc64_encoder_data(encoder.context), size)
    stream = store / 'tokens.rc64'
    jg2.persist_immutable_bytes(stream, raw, label='TC1 raw stream')
    rider = codec.MAGIC + weight_path.read_bytes() + raw
    jg2.persist_immutable_bytes(store / 'tail_rider.bin', rider, label='TC1 counted rider')
    result = dict(axis=AXIS, score_claim=False, pairs=codec.N, symbols=codec.N * codec.H * codec.W,
                  tag=tag, archive=inputs['archive'], stream=jg2.file_fact(stream),
                  rider=jg2.file_fact(store / 'tail_rider.bin'), weights=jg2.file_fact(weight_path),
                  exact_frequency_ideal_bytes=float(per_frame.sum() / 8),
                  all600_feature_trace_identity=True, resumed_from_frame=start,
                  build=build, elapsed_s=time.monotonic() - started,
                  checkpoint=jg2.file_fact(store / 'state_0600.npz'))
    jg2.atomic_json(store / 'ENCODE.json', result)
    return result


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'expected one reader rewrite: {old[:80]!r}')
    return text.replace(old, new)


def stage(root):
    """Retain the container race, then patch only the tail reader and archive pins."""
    import brotli
    inputs, _ = admitted(root)
    a = root / 'mixer/primary/tokens.rc64'
    b = root / 'mixer/repeat/tokens.rc64'
    if a.read_bytes() != b.read_bytes():
        raise RuntimeError('TC1 twin encodes differ')
    for tag in ('primary', 'repeat'):
        result = json.loads((root / 'mixer' / tag / 'ENCODE.json').read_text())
        if result['pairs'] != codec.N or not result['all600_feature_trace_identity']:
            raise RuntimeError('incomplete encoder receipt')
    section = jg2.split_member(jg2.read_archive_member(Path(inputs['archive']['path'])))
    header = bytearray(section['header']); header[7] |= 0x80
    prefix = bytes(header) + section['hpac'] + section['semantic'] + section['carrier'] + section['tail'][:96]
    weights, raw = (root / 'mixer/weights_i8.bin').read_bytes(), a.read_bytes()
    store = root / 'mixer/containers'; store.mkdir(exist_ok=True)
    alternatives = [('plain', codec.MAGIC + weights + raw)]
    for quality in (9, 10, 11):
        for window in (22, 23, 24):
            packed = brotli.compress(raw, quality=quality, lgwin=window)
            rider = codec.MAGIC[:4] + b'\x11' + weights + packed
            jg2.persist_immutable_bytes(store / f'brotli_q{quality}_w{window}.bin', rider, label='TC1 Brotli rider')
            if codec.unpack_rider(rider) != (weights, raw):
                raise RuntimeError('TC1 container parse-back failed')
            alternatives.append((f'brotli_q{quality}_w{window}', rider))
    census = []
    for label, rider in alternatives:
        rider_path = store / (label + '.bin')
        jg2.persist_immutable_bytes(rider_path, rider, label='TC1 container rider')
        member = prefix + rider
        for method, level in [('stored', None), ('deflate', 1), ('deflate', 6), ('deflate', 9)]:
            archive = store / f'{label}_{method}_{level}.zip'
            if method == 'stored':
                jg2.pack_archive(member, archive)
            else:
                with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=level) as z:
                    info = zipfile.ZipInfo('p', date_time=jg2.SHIPPED_ZIP_DATE_TIME)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = jg2.SHIPPED_ZIP_EXTERNAL_ATTR
                    info.create_system = jg2.SHIPPED_ZIP_CREATE_SYSTEM
                    z.writestr(info, member, compresslevel=level)
            if jg2.read_archive_member(archive) != member:
                raise RuntimeError('ZIP parse-back changed member p')
            census.append(dict(label=label, method=method, level=level, archive=jg2.file_fact(archive), rider=jg2.file_fact(rider_path)))
    winner = min(census, key=lambda x: (x['archive']['bytes'], x['label'] != 'plain', x['method'] != 'stored', str(x['level'])))
    runtime = root / 'candidate_runtime'
    if runtime.exists():
        receipt = root / 'mixer/RESULT.json'
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            if prior['candidate'] == jg2.file_fact(runtime / 'archive.zip'):
                return prior
        raise RuntimeError('incomplete candidate tree requires inspection')
    runtime = root / 'candidate_runtime_build'
    if runtime.exists():
        raise RuntimeError('partial candidate build retained; inspect before resuming')
    shutil.copytree(root / 'source_runtime', runtime, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.so', '*.dylib'))
    shutil.copyfile(winner['archive']['path'], runtime / 'archive.zip')
    shutil.copyfile(REPO / 'experiments/ddm_tc1_mixer_codec.py', runtime / 'runtime/tc1_shared_mixer.py')
    shutil.copyfile(REPO / 'experiments/ddm_tc1_receiver_checkpoint.py', runtime / 'runtime/tc1_receiver_checkpoint.py')
    path = runtime / 'runtime/residual_archive.py'
    text = path.read_text()
    text = _replace_once(text, 'SZ1_RESERVED_KNOWN_BITS = 0x7F', 'SZ1_RESERVED_KNOWN_BITS = 0xFF')
    text = _replace_once(text, '    token_codec: str = "rc64"', '    token_codec: str = "rc64"\n    tc1_weights: bytes | None = None')
    text = _replace_once(text, '    tokens = section[compact_size:]\n', '    tokens = section[compact_size:]\n    tc1_weights = None\n    if rx1 is not None and outer[7] & 0x80:\n        from .tc1_shared_mixer import unpack_rider\n        tc1_weights, tokens = unpack_rider(tokens)\n')
    text = _replace_once(text, '        token_stream=tokens,', '        token_stream=tokens,\n        tc1_weights=tc1_weights,')
    text = _replace_once(text, '    decoder = NativeDecoder(Path(library), parts.token_stream)', '    decoder = NativeDecoder(Path(library), parts.token_stream)\n    from .tc1_shared_mixer import SharedMixer\n    mixer = None if parts.tc1_weights is None else SharedMixer(parts.tc1_weights)')
    text = _replace_once(text, '            corrector.begin_frame(boundary)', '            corrector.begin_frame(boundary)\n            previous_field = None if not frame else previous[0].to(device="cpu", dtype=torch.uint8).numpy()\n            if mixer is not None:\n                mixer.begin_frame()')
    text = _replace_once(text, '                symbols = decoder.decode(\n                    corrector.coding_row(state)\n                ).astype(np.int64)', '                coding = corrector.coding_row(state)\n                if mixer is not None:\n                    partial = current[0].to(device="cpu", dtype=torch.uint8).numpy()\n                    coding = mixer.coding(coding, flat_positions, partial, previous_field)\n                symbols = decoder.decode(coding).astype(np.int64)')
    text = _replace_once(text, '            corrector.end_frame(tokens[frame].numpy().reshape(-1))', '            corrector.end_frame(tokens[frame].numpy().reshape(-1))\n            if mixer is not None:\n                mixer.end_frame(tokens[frame].numpy(), previous_field)')
    text = _replace_once(text, '        for frame in range(runtime.N):', '        checkpoint = None\n        resumed_from = 0\n        if mixer is not None and os.environ.get("TC1_RECEIVER_CHECKPOINT_DIR"):\n            from .tc1_receiver_checkpoint import ReceiverCheckpoint\n            checkpoint = ReceiverCheckpoint(parts, decoder, corrector, mixer, code_dir)\n            resumed_from = checkpoint.restore(tokens, previous)\n        for frame in range(resumed_from, runtime.N):')
    text = _replace_once(text, '            previous = current\n', '            previous = current\n            if checkpoint is not None and ((frame + 1) % 25 == 0 or frame + 1 == int(os.environ.get("TC1_RECEIVER_STOP_AFTER", "600"))):\n                checkpoint.save(frame + 1, tokens)\n')
    text = _replace_once(text, '        "token_codec": "rc64",', '        "token_codec": "rc64",\n        "tail_mixer": "none" if mixer is None else "TC1M-35-int8",')
    text = _replace_once(text, '        "free_corrector": _rr8_corrector_kind(corrector),', '        "free_corrector": _rr8_corrector_kind(corrector),\n        "checkpoint_resumed_from_frame": resumed_from,\n        "logit_and_cdf_digest_scope": "suffix from checkpoint_resumed_from_frame; token digest is full n600",')
    compile(text, str(path), 'exec'); path.write_text(text)
    entry = runtime / 'inflate.py'; text = entry.read_text()
    text = _replace_once(text, inputs['archive']['sha256'], winner['archive']['sha256'])
    text = _replace_once(text, f"ARCHIVE_BYTES = {inputs['archive']['bytes']}", f"ARCHIVE_BYTES = {winner['archive']['bytes']}")
    entry.write_text(text)
    changed = []
    for p in sorted(runtime.rglob('*')):
        if p.is_file():
            relative = p.relative_to(runtime); old = root / 'source_runtime' / relative
            if not old.exists() or old.read_bytes() != p.read_bytes():
                changed.append(str(relative))
    expected = ['archive.zip', 'inflate.py', 'runtime/residual_archive.py', 'runtime/tc1_receiver_checkpoint.py', 'runtime/tc1_shared_mixer.py']
    if changed != expected:
        raise RuntimeError(f'unexpected candidate source census: {changed}')
    runtime.rename(root / 'candidate_runtime')
    runtime = root / 'candidate_runtime'
    result = dict(axis=AXIS, score_claim=False, source=inputs['archive'],
                  twins_byte_identical=True, candidate=jg2.file_fact(runtime / 'archive.zip'),
                  runtime=str(runtime), container_sweep=census, selected=winner,
                  net_saved_bytes=inputs['archive']['bytes'] - winner['archive']['bytes'],
                  changed_paths=changed, changed_archive_sections=['header tail-codec flag', 'tail'],
                  hpac_semantic_carrier_byte_identical=True, public_decode_identity='PENDING')
    jg2.atomic_json(root / 'mixer/RESULT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['quantize', 'verify', 'encode', 'stage'])
    parser.add_argument('--root', type=Path, default=ROOT / 'rebase_pc2')
    parser.add_argument('--tag', choices=['primary', 'repeat'], default='primary')
    args = parser.parse_args()
    result = {'quantize': lambda: quantize(args.root), 'verify': lambda: verify_codec(args.root), 'encode': lambda: encode(args.root, args.tag), 'stage': lambda: stage(args.root)}[args.stage]()
    print(json.dumps(result, sort_keys=True))
