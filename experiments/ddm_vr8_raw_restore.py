"""Restore an SJ1 raw from its actual receiver and retained token-stage checkpoint.

Only the receiver's render writer is intercepted in memory. Token decoding is
forbidden: the archive/runtime-bound complete checkpoint must already exist.
The CPU batch-one equations and original selector produce each retained pair.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import ddm_vr7_rebuild as vr7
from experiments import ddm_vr8_restore as common


class Captured(Exception):
    """Private control transfer after receiver input loading and before rendering."""


def copy_pinned(expected: dict, destination: Path) -> None:
    """Copy to a new owned path; verify rather than overwrite on resume."""
    common.go()
    source = Path(expected['path'])
    common.safe_path(source)
    common.safe_path(destination)
    if vr7.fact(source) != expected:
        raise ValueError('retained input drift')
    if not destination.exists():
        with source.open('rb') as src, destination.open('xb') as dst:
            while chunk := src.read(1 << 20):
                common.go()
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
    measured = vr7.fact(destination)
    if (measured['bytes'], measured['sha256']) != (expected['bytes'], expected['sha256']):
        raise ValueError('copied or interrupted input differs; retained without overwrite')


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError('receiver module cannot be loaded')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def capture_receiver(source: dict, root: Path, inputs: dict):
    """Run actual F26 setup/checkpoint loading; capture before any raw is written."""
    common.go()
    runtime = Path(source['runtime'])
    archive = Path(source['archive'])
    checkpoint = root / 'receiver_checkpoint'
    build = root / 'build'
    data = root / 'data'
    for directory in (checkpoint, build, data):
        common.safe_path(directory)
        directory.mkdir(exist_ok=True)
    for key, target in [('tokens', checkpoint / 'tokens_cpu_stage_complete.u8'),
                        ('token_receipt', checkpoint / 'tokens_cpu_stage_complete.json'),
                        ('rc64', build / 'rc64_backend.so'),
                        ('corrector', build / 'f26_corrector_native.so')]:
        copy_pinned(inputs[key], target)
    with zipfile.ZipFile(archive) as zf:
        payload = zf.read('p')
    extracted = data / 'p'
    common.safe_path(extracted)
    if extracted.exists():
        if extracted.read_bytes() != payload:
            raise ValueError('extracted archive member drift; no overwrite')
    else:
        with extracted.open('xb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    os.environ['CPR1_RC64_LIBRARY'] = str(build / 'rc64_backend.so')
    os.environ['F26_CORRECTOR_NATIVE_LIBRARY'] = str(build / 'f26_corrector_native.so')
    os.environ['F26_TOKEN_DECODER'] = 'python'
    for key in ('F26_ADVISORY_RENDER_WORKERS', 'F26_ADVISORY_DECODE_CACHE_ROOT',
                'F26_ADVISORY_PAIR_LIMIT', 'TC1_RECEIVER_CHECKPOINT_DIR', 'TC1_RECEIVER_STOP_AFTER'):
        os.environ.pop(key, None)
    sys.path.insert(0, str(runtime))
    receiver = load_module(runtime / 'inflate.py', '_vr8_raw_entry')
    receiver._verify_input(data, archive)
    f26 = importlib.import_module('runtime.f26_inflate')
    if Path(f26.__file__).resolve() != runtime / 'runtime/f26_inflate.py':
        raise ValueError('foreign F26 runtime already imported; use one process per raw')
    renderer = f26._load_renderer(runtime / 'cpr1')
    state = {}

    def capture(semantic, basis, coefficients, tokens, destination, device):
        state.update(semantic=semantic, basis=basis, coefficients=coefficients,
                     tokens=tokens, device=device)
        raise Captured()

    def refuse_decode(*args, **kwargs):
        raise ValueError('retained token checkpoint did not load; decoding forbidden')

    original_render, original_decode = renderer.render_video, f26.decode_production_tokens
    renderer.render_video = capture
    f26.decode_production_tokens = refuse_decode
    try:
        f26.inflate_archive(archive, root / 'NEVER_WRITTEN.raw', renderer_dir=runtime / 'cpr1',
                            device_name='cpu', num_threads=4, checkpoint_dir=checkpoint)
    except Captured:
        pass
    finally:
        renderer.render_video, f26.decode_production_tokens = original_render, original_decode
    if not state or (root / 'NEVER_WRITTEN.raw').exists():
        raise ValueError('receiver capture did not stop before raw output')
    parts = f26.read_residual_archive(archive)
    _, selector_blob = f26.split_frame0_selector_carrier(parts.carrier_blob)
    selector = None if selector_blob is None else f26.decode_selector(selector_blob)
    if selector is not None and len(selector[1]) != renderer.N:
        raise ValueError('selector count differs')
    return renderer, f26, state, selector


def pair_renderer(renderer, f26, state: dict, selector):
    """Exact CPU batch-one operations from the captured receiver render_video."""
    import torch

    device = state['device']
    if device.type != 'cpu':
        raise ValueError('raw restore requires CPU')
    semantic = state['semantic'].eval().to(device)
    basis = renderer.normalized_basis(state['basis'].to(device))
    coefficients = state['coefficients'].to(device)
    tokens = state['tokens']

    def pair(index: int) -> bytes:
        with torch.no_grad():
            indices = torch.arange(index, index + 1, device=device)
            master = (torch.nn.functional.interpolate(
                semantic(tokens[index:index + 1].long().to(device), indices),
                size=(renderer.CAMERA_H, renderer.CAMERA_W), mode='bilinear', align_corners=False)
                .clamp(0.0, 255.0).round())
            master_np = master.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            carrier = torch.einsum('bk,kchw->bchw', coefficients[index:index + 1], basis)
            carrier = carrier / math.sqrt(renderer.CARRIER_DIM)
            slave = (torch.nn.functional.interpolate(
                (127.5 + renderer.CARRIER_AMPLITUDE * carrier).clamp(0.0, 255.0).round(),
                size=(renderer.CAMERA_H, renderer.CAMERA_W), mode='bicubic', align_corners=False)
                .clamp(0.0, 255.0).round())
            slave_np = slave.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            if selector is not None:
                modes, indices = selector
                mode_index = int(indices[index])
                slave_np = f26.apply_pixel_mode(slave_np.copy(), modes[mode_index])
            return slave_np.tobytes() + master_np.tobytes()

    return pair


def write_pairs(source: dict, output: Path, root: Path, binding: dict, render, pair_bytes: int) -> int:
    if pair_bytes * 600 != source['expected_bytes']:
        raise ValueError('n600 raw geometry differs')
    start = common.prefix(output, root, binding, list(range(600)), pair_bytes)
    common.reserve(output, (600 - start) * pair_bytes)
    if not output.exists():
        with output.open('xb'):
            pass
    with output.open('ab') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for slot in range(start, 600):
            common.go()
            common.reserve(output, (600 - slot) * pair_bytes)
            payload = render(slot)
            if len(payload) != pair_bytes or output.stat().st_size != slot * pair_bytes:
                with (root / f'blocked_pair_{slot:04d}.bin').open('xb') as retained:
                    retained.write(payload)
                    retained.flush()
                    os.fsync(retained.fileno())
                raise ValueError('pair geometry/output drift; materialized payload retained')
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            common.persist(root / f'frame_{slot:04d}.json', {
                'binding': binding, 'pair': slot, 'slot': slot, 'bytes': pair_bytes,
                'sha256': hashlib.sha256(payload).hexdigest()})
    common.go()
    vr7.verify(source)
    rebuilt = vr7.fact(output)
    matches = (rebuilt['bytes'], rebuilt['sha256']) == (source['expected_bytes'], source['expected_sha256'])
    result = {'schema': 'ddm_vr8.raw_restore_certificate.v1', 'complete': matches,
              'kind': 'sj1_parseback', 'config': binding, 'rebuilt': rebuilt,
              'score_claim': False, 'consumer_store': binding['consumer_store'],
              'mechanism': 'actual receiver setup and token checkpoint; CPU batch-one pair writer',
              'hygiene': 'all outputs and checkpoints retained; no cleanup', 'seed': 20260910, 'threads': 4}
    path = root / 'RESTORE_CERTIFICATE.json'
    common.safe_path(path)
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError('raw certificate drift')
    else:
        common.persist(path, result)
    return 0 if matches else 3


def run(config_path: Path) -> int:
    common.safe_path(config_path)
    config = json.loads(config_path.read_text())
    if (config.get('schema') != 'ddm_vr8.raw_restore.v1' or config.get('seed') != 20260910
            or config.get('threads') != 4 or not config.get('consumer_store')):
        raise ValueError('pinned raw restore configuration required')
    root, output = Path(config['checkpoint_dir']), Path(config['output'])
    common.safe_path(root)
    common.safe_path(output)
    if root.parent != common.RETAINED or root.name.startswith('.'):
        raise ValueError('private VR8 retained root required')
    common.go()
    expected = config['source_config']
    source_path = Path(expected['path'])
    common.safe_path(source_path)
    if vr7.fact(source_path) != expected:
        raise ValueError('source config drift')
    source = json.loads(source_path.read_text())
    if str(output) != source['original'] or not str(output).startswith(
            '/Volumes/APDataStore/pact/cold_store_sj1_bulk_20260910/') or output.name != '0.raw':
        raise ValueError('exact missing historical raw destination required')
    vr7.verify(source)
    common.reserve(output, max(0, source['expected_bytes'] - (output.stat().st_size if output.exists() else 0)))
    root.mkdir(parents=True, exist_ok=True)
    # Token copies are charged to the metadata SSD tier as well as output reserve.
    common.reserve(root / 'input_budget', sum(v['bytes'] for v in config['retained_inputs'].values()))
    binding = {'config': vr7.fact(config_path), 'runner': vr7.fact(Path(__file__).absolute()),
               'writer': vr7.fact(Path(common.__file__).absolute()), 'source_config': expected,
               'consumer_store': config['consumer_store']}
    lock = root / 'LOCK'
    common.safe_path(lock)
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / 'BINDING.json'
        common.safe_path(path)
        if path.exists():
            if json.loads(path.read_text()) != binding:
                raise ValueError('raw resume binding drift')
        elif output.exists():
            raise ValueError('unowned raw destination exists')
        else:
            common.persist(path, binding)
        random.seed(20260910)
        import numpy as np
        import torch
        np.random.seed(20260910)
        torch.manual_seed(20260910)
        torch.use_deterministic_algorithms(True)
        renderer, f26, state, selector = capture_receiver(source, root, config['retained_inputs'])
        if renderer.N != 600:
            raise ValueError('original n600 renderer required')
        return write_pairs(source, output, root, binding,
                           pair_renderer(renderer, f26, state, selector),
                           2 * renderer.CAMERA_H * renderer.CAMERA_W * 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', type=Path, required=True)
    args = parser.parse_args()
    return run(args.resume_from.absolute())


if __name__ == '__main__':
    raise SystemExit(main())
