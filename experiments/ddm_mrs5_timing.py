"""Source-bound sampled A/B/A/B diagnostic; never a cold-n600/public-path receipt.

Both original shells execute their compiler/setup work. An external PATH shim
intercepts only the final exact inflate.py argv. A retains its sealed token-loop
arithmetic and renderer expressions through AST extraction; B calls TokenDecoder.
Every pair restores the same retained causal state. Process-cold is not n600-cold.
Interrupted sequences retain all bytes and restart the entire four-run sequence.
"""
from __future__ import annotations

import argparse
import ast
import copy
import ctypes
import fcntl
import gzip
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pickle
import platform
import random
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
import zipfile

ROOT = Path('/Volumes/APDataStore/pact/ddm_mrs5')
SEALED = Path('/Volumes/APDataStore/pact/ddm_pd6/candidate2/candidate_runtime')
CANDIDATE = ROOT / 'submissions/mrs5'
BANK = Path('/Volumes/APDataStore/pact/ddm_mrs2/profile_python/RESULT.json')
LEG = Path('/Volumes/APDataStore/pact/ddm_pd6/SEAL_ddm_pd6_price_first_generator_contest_cuda.json.decode_wall_clock.json')
REPO = Path(__file__).resolve().parents[1]
SEED = 20260916
ARCHIVE_SHA = 'aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957'
SCOPE = ('cold process and own public shell native builds; intercepted Python entrypoint; '
         '48 seeded stratified pairs restored from pre-pair checkpoints; includes model setup, '
         'state load, token decode, own renderer, selector, payload compression/verification and writes; '
         'excludes archive extraction and source preflight; NOT cold n600, NOT an unmodified public replay')


def fact(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.pending')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def source_files(root):
    return [fact(p) for p in sorted(root.rglob('*')) if p.is_file()
            and '__pycache__' not in p.parts and not p.name.startswith('.')
            and p.suffix not in {'.pyc', '.so', '.dylib'}]


def verify(ref):
    if fact(ref['path']) != ref:
        raise ValueError('source or payload drift: ' + ref['path'])


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def original_function(module, name):
    source = inspect.getsource(getattr(module, name))
    node = ast.parse(source).body[0]
    if not isinstance(node, ast.FunctionDef):
        raise ValueError('expected a source function')
    return source, node


def compile_function(module, node, label, directory):
    node = ast.fix_missing_locations(node)
    text = ast.unparse(node) + '\n'
    path = directory / (label + '.py.txt')
    path.write_text(text)
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[node.name], fact(path)


def sampled_sealed_setup(module, directory):
    source, node = original_function(module, 'inflate_archive')
    stop = [i for i, statement in enumerate(node.body) if isinstance(statement, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'setup_seconds' for t in statement.targets)]
    if len(stop) != 1:
        raise ValueError('sealed archive setup boundary changed')
    node.body = node.body[:stop[0] + 1] + ast.parse('return locals()').body
    (directory / 'sealed_setup.original.py.txt').write_text(source)
    return compile_function(module, node, 'sealed_setup.extracted', directory)


def sampled_sealed_decoder(module, frames, restore_pair, save_token, directory):
    source, node = original_function(module, 'decode_production_tokens')
    contexts = [s for s in node.body if isinstance(s, ast.With)]
    if len(contexts) != 1:
        raise ValueError('sealed inference context changed')
    context = contexts[0]
    loops = [s for s in context.body if isinstance(s, ast.For)
             and isinstance(s.target, ast.Name) and s.target.id == 'frame']
    if len(loops) != 1 or ast.unparse(loops[0].iter) != 'range(resumed_from, runtime.N)':
        raise ValueError('sealed frame loop changed')
    loop = loops[0]
    loop.iter = ast.Name(id='_mrs5_frames', ctx=ast.Load())
    loop.body = ast.parse('previous = _mrs5_restore(frame, decoder, corrector, mixer)\n_mrs5_pair_started = time.perf_counter()').body + loop.body
    loop.body += ast.parse('_mrs5_save(frame, tokens[frame], time.perf_counter() - _mrs5_pair_started)').body
    token_assignments = [s for s in context.body if isinstance(s, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'tokens' for t in s.targets)]
    if len(token_assignments) != 1:
        raise ValueError('sealed token allocation changed')
    token_assignments[0].value = ast.Dict(keys=[], values=[])
    node.body = node.body[:node.body.index(context) + 1] + ast.parse('return tokens').body
    (directory / 'sealed_token_loop.original.py.txt').write_text(source)
    module._mrs5_frames, module._mrs5_restore, module._mrs5_save = frames, restore_pair, save_token
    return compile_function(module, node, 'sealed_token_loop.extracted', directory)


def render_expressions(module, name, directory):
    """Keep each receiver's own tensor arithmetic; remove full-field output allocation."""
    source, node = original_function(module, name)
    loops = [s for s in node.body if isinstance(s, ast.For)]
    if len(loops) != 2 or any(not isinstance(s.target, ast.Name) or s.target.id != 'start' for s in loops):
        raise ValueError('renderer loop structure changed')
    # These first three assignments are model/device, basis normalization, coefficients/device.
    if [ast.unparse(s.targets[0]) for s in node.body[:3] if isinstance(s, ast.Assign)] != ['semantic', 'basis', 'coefficients']:
        raise ValueError('renderer setup changed')
    prefix = copy.deepcopy(node.body[:3])
    bodies = []
    for loop in loops:
        inner = next((i for i, s in enumerate(loop.body) if isinstance(s, ast.For)), None)
        if inner is None:
            raise ValueError('renderer output loop missing')
        bodies.extend(copy.deepcopy(loop.body[:inner]))
    arguments = 'semantic, basis, coefficients, tokens, device, start'
    function = ast.parse(f'def sampled_render({arguments}):\n    pass\n').body[0]
    function.body = ast.parse('semantic_batch = pose_batch = 1').body + bodies + ast.parse('return master_np, slave_np').body
    setup = ast.parse('def renderer_setup(semantic, basis, coefficients, device):\n    pass\n').body[0]
    setup.body = prefix + ast.parse('return semantic, basis, coefficients').body
    (directory / 'renderer.original.py.txt').write_text(source)
    renderer, render_ref = compile_function(module, function, 'renderer.extracted', directory)
    initialize, setup_ref = compile_function(module, setup, 'renderer_setup.extracted', directory)
    return initialize, renderer, [render_ref, setup_ref]


def translate_state(value):
    """Only rename the four module-prefix class names changed by the minimal port."""
    names = {'geometry_mixer_LaneMixer': 'LaneMixer', 'shared_SharedMixer': 'SharedMixer',
             'miss_FreeCorrector': 'FreeCorrector', 'odds_MixerFamily': 'MixerFamily'}
    kind, payload = value
    if kind == 'object':
        name, members = payload
        if name not in names:
            raise ValueError('unreviewed sealed state class: ' + name)
        return kind, (names[name], {k: translate_state(v) for k, v in members.items()})
    if kind == 'dict':
        return kind, {k: translate_state(v) for k, v in payload.items()}
    if kind in {'list', 'tuple'}:
        return kind, [translate_state(v) for v in payload]
    return value


def retained_blob(path, payload):
    """Persist first, then prove lossless decode before the in-memory bytes leave scope."""
    raw = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    with path.open('xb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', compresslevel=1, mtime=0) as compressed:
            compressed.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    with gzip.open(path, 'rb') as stream:
        restored = stream.read()
    if restored != payload:
        raise ValueError('retained compressed payload failed exact readback')
    return dict(uncompressed=raw, retained=fact(path), encoding='gzip level 1', lossless_readback=True)


def worker(run, arm, runtime, limit):
    import numpy as np
    import torch
    from ddm_mrs1_validate import restore
    from ddm_mrs4_proof import native_arrays, native_record, restore_pair

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = json.loads(BANK.read_text())['rows'][:limit]
    frames = [row['frame'] for row in rows]
    lookup = {row['frame']: row for row in rows}
    decoded, receipts = {}, {}
    scope = SCOPE if limit == 48 else SCOPE.replace('48 seeded stratified pairs', 'one bank pair (instrument smoke only)')
    instruction = dict(axis='[macOS-CPU advisory]', scope=scope, score_claim=False,
        process_cold=True, cold_n600=False, per_pair_checkpoint_restore=True, sample=frames,
        torch=torch.__version__, numpy=np.__version__, threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(), instrument=fact(Path(__file__)))
    atomic(run / 'INSTRUMENTATION.json', instruction)

    def save_token(frame, token, token_seconds):
        payload = token.numpy().tobytes()
        stored = retained_blob(run / f'pair_{frame:04d}.tokens.u8.gz', payload)
        receipts[frame] = dict(frame=frame, state=lookup[frame]['state'], token=stored, token_seconds=token_seconds)
        atomic(run / f'pair_{frame:04d}.json', receipts[frame])
        if stored['uncompressed']['sha256'] != lookup[frame]['token']['sha256']:
            raise ValueError('token differs from retained bank; materialized bytes kept')
        decoded[frame] = token.clone()

    with torch.inference_mode():
        if arm == 'A':
            sys.path.insert(0, str(runtime))
            from runtime import f26_inflate as inflate
            from runtime import residual_archive as residual
            from runtime.free_corrector import FreeCorrector
            from runtime.tc1_receiver_checkpoint import CorrectorPrefix, DecoderState
            compiled = run / 'native'
            compiled.mkdir()
            libraries = {}
            for key in ('CPR1_RC64_LIBRARY', 'F26_CORRECTOR_NATIVE_LIBRARY', 'RLC1_GEOMETRY_LIBRARY'):
                loaded = Path(os.environ[key])
                retained = compiled / loaded.name
                shutil.copy2(loaded, retained)
                libraries[key] = fact(retained)
                if fact(loaded)['sha256'] != libraries[key]['sha256']:
                    raise ValueError('retained native library differs from the loaded library')
            if os.environ.get('F26_TOKEN_DECODER') != 'python' or 'F26_HPAC_NATIVE_LIBRARY' in os.environ:
                raise ValueError('sealed measured native set changed')
            # Execute input verification from the immutable public entrypoint itself.
            entry = load_module(runtime / 'inflate.py', 'mrs5_sealed_entry')
            entry._verify_input(run / 'data', runtime / 'archive.zip')
            setup, setup_ref = sampled_sealed_setup(inflate, run)
            context = setup(runtime / 'archive.zip', run / 'unused.raw', renderer_dir=runtime / 'cpr1',
                            device_name='cpu', num_threads=4, checkpoint_dir=run / 'unused_checkpoints')
            parts, renderer = context['parts'], context['renderer']

            def sealed_restore(frame, decoder, corrector, mixer):
                row = lookup[frame]
                verify(row['state'])
                with gzip.open(row['state']['path'], 'rb') as stream:
                    state = pickle.load(stream)  # Trusted local, exact SHA-bound bank only.
                if state['frame'] != frame or type(corrector).__name__ != 'NativeFreeCorrector':
                    raise ValueError('sealed restoration backend or frame mismatch')
                reference = FreeCorrector(384 * 512)
                restore(reference, translate_state(state['corrector']))
                record = ctypes.cast(corrector.handle, ctypes.POINTER(CorrectorPrefix)).contents
                record.have_prev = reference.have_prev
                for _, target, values in native_arrays(record, reference):
                    target[:] = values
                restore(mixer, translate_state(state['mixer']))
                range_state = state['range'][1][1]
                if range_state['payload'][1] != parts.token_stream:
                    raise ValueError('range state belongs to a different token stream')
                decoder_state = ctypes.cast(decoder.context, ctypes.POINTER(DecoderState)).contents
                for field in ('low', 'high', 'code', 'bit_position'):
                    setattr(decoder_state, field, int(range_state[field][1]))
                decoder_state.error = 0
                if mixer.frame != frame or decoder_state.bit_position != range_state['bit_position'][1]:
                    raise ValueError('restored causal phase differs')
                return torch.from_numpy(state['previous']).to('cpu')

            decode, decode_ref = sampled_sealed_decoder(residual, frames, sealed_restore, save_token, run)
            decode(parts, renderer, runtime / 'cpr1', torch.device('cpu'))
            initialize, render, render_refs = render_expressions(renderer, 'render_video', run)
            semantic, basis, coefficients = initialize(context['semantic'], context['basis'], context['coefficients'], torch.device('cpu'))
            modes, labels = inflate.decode_selector(context['selector_blob'])
            select = inflate.apply_pixel_mode
            instruction.update(native_libraries=libraries, extracted=[setup_ref, decode_ref, *render_refs],
                edits=['setup cut after setup_seconds; return locals',
                       'frame iterator is exact bank sample; inject causal state restore',
                       'tokens uses selected-frame dict; emit retained token after each original frame body',
                       'drop n600-only report tail; no fake full-field report',
                       'renderer keeps original setup and batch-one tensor expressions; sample index supplied; output written externally'])
        else:
            module = load_module(runtime / 'inflate.py', 'mrs5_timed_receiver')
            if any(getattr(module, key) is None for key in ('_RANGE_LIBRARY', '_CORRECTOR_LIBRARY', '_GEOMETRY_LIBRARY')):
                raise ValueError('candidate did not load all three native libraries')
            libraries = {name: fact(runtime / (name + '.so')) for name in ('range_decoder', 'corrector', 'geometry')}
            with zipfile.ZipFile(runtime / 'archive.zip') as archive:
                if archive.namelist() != ['p'] or archive.read('p') != (run / 'data/p').read_bytes():
                    raise ValueError('candidate extracted payload differs from archive')
            parts, model, basis, coefficients, selector = module.read_models(runtime / 'archive.zip')
            decoder = module.TokenDecoder(parts, torch.device('cpu'))
            if not isinstance(decoder.corrector, module.NativeCorrector) or module.geometry_mixer_LaneGeometry is not module.NativeGeometry:
                raise ValueError('candidate silently selected a Python native-set fallback')
            record_type = native_record((runtime / 'corrector.c').read_text())
            for row in rows:
                verify(row['state'])
                restore_pair(Path(row['state']['path']), decoder, module, record_type)
                if decoder.frame != row['frame']:
                    raise ValueError('candidate restored frame differs')
                started = time.perf_counter()
                token = next(decoder)
                save_token(row['frame'], token, time.perf_counter() - started)
            decoder.corrector.close()
            initialize, render, render_refs = render_expressions(module, 'render_render_video', run)
            semantic, basis, coefficients = initialize(model, basis, coefficients, torch.device('cpu'))
            modes, labels = module.selector_decode_selector(selector)
            select = module.selector_apply_pixel_mode
            instruction.update(native_libraries=libraries, extracted=render_refs,
                edits=['TokenDecoder unchanged; external restore_pair before each selected pair',
                       'renderer keeps original setup and batch-one tensor expressions; sample index supplied; output written externally'])

        class SelectedTokens:
            def __getitem__(self, item):
                if not isinstance(item, slice) or item.stop != item.start + 1 or item.step is not None:
                    raise ValueError('sample renderer requested unsupported token extent')
                return decoded[item.start][None]

        for frame in frames:
            started = time.perf_counter()
            master, slave = render(semantic, basis, coefficients, SelectedTokens(), torch.device('cpu'), frame)
            slave = select(slave.copy(), modes[int(labels[frame])])
            receipts[frame]['render_selector_seconds'] = time.perf_counter() - started
            raw = np.stack((slave[0], master[0])).tobytes()
            receipts[frame]['raw'] = retained_blob(run / f'pair_{frame:04d}.raw.gz', raw)
            match = receipts[frame]['raw']['uncompressed']['sha256'] == lookup[frame]['raw']['sha256']
            receipts[frame]['matches_bank'] = match
            atomic(run / f'pair_{frame:04d}.json', receipts[frame])
            if not match:
                raise ValueError('raw differs from retained bank; materialized bytes kept')
        instruction['original_extractions'] = [fact(p) for p in sorted(run.glob('*.original.py.txt'))]
        atomic(run / 'INSTRUMENTATION.json', instruction)
        atomic(run / 'WORKER_RESULT.json', dict(instruction=instruction, pair_count=len(frames),
            pairs=[receipts[frame] for frame in frames], completed=True, score_claim=False,
            pair_compute_seconds=sum(r['token_seconds'] + r['render_selector_seconds'] for r in receipts.values()),
            pair_compute_scope='token loop plus own renderer/selector; excludes model setup, state restoration, compression, hashes and output IO'))


def uptime():
    return dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                load_average=list(os.getloadavg()),
                uptime=subprocess.check_output(['/usr/bin/uptime'], text=True).strip())


def group_exists(group):
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    return True


def terminate_group(child):
    """A dead shell leader does not prove its worker/grandchildren stopped."""
    sent = []
    for number, grace in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(child.pid, number)
            sent.append(number)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            child.poll()  # Reap our direct child independently of group liveness.
            if not group_exists(child.pid):
                break
            time.sleep(0.05)
        if not group_exists(child.pid):
            break
    child.wait(timeout=5)
    return dict(signals=sent, direct_child_reaped=True,
                process_group_absent=not group_exists(child.pid))


def execute_shell(argv, environment, lock_fd, stdout, stderr, timeout=1800):
    """Retain control on supervisor signals; return failure so RUN can be saved first."""
    numbers = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)
    child, pending, failure, cleanup = None, [], None, None

    def stop(number, frame):
        pending.append(number)
        # A signal during Popen construction is handled immediately after assignment.
        if child is not None:
            raise InterruptedError(f'interrupted by signal {number}')

    previous = {number: signal.signal(number, stop) for number in numbers}
    started = time.perf_counter()
    try:
        child = subprocess.Popen(argv, env=environment, stdout=stdout, stderr=stderr,
                                 start_new_session=True, pass_fds=(lock_fd,))
        if pending:
            raise InterruptedError(f'interrupted during child launch by signal {pending[-1]}')
        child.wait(timeout=timeout)
    except BaseException as error:
        failure = error
        for number in numbers:
            signal.signal(number, signal.SIG_IGN)
        if child is not None:
            try:
                cleanup = terminate_group(child)
            except BaseException as cleanup_error:
                cleanup = dict(exception=repr(cleanup_error), direct_child_reaped=child.poll() is not None,
                               process_group_absent=not group_exists(child.pid))
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    return dict(returncode=None if child is None else child.returncode,
                wall_seconds=time.perf_counter() - started,
                child_pid=None if child is None else child.pid,
                exception=None if failure is None else repr(failure), cleanup=cleanup), failure


def binding(limit):
    import numpy as np
    bank = json.loads(BANK.read_text())
    randomizer = np.random.default_rng(SEED)
    frames = [int(i) for start in range(0, 600, 25)
              for i in sorted(randomizer.choice(25, 2, replace=False) + start)]
    if [r['frame'] for r in bank['rows']] != frames:
        raise ValueError('bank is not the charter seeded stratified 48')
    for row in bank['rows'][:limit]:
        for key in ('state', 'token', 'raw'):
            verify(row[key])
    for runtime in (SEALED, CANDIDATE):
        if fact(runtime / 'archive.zip')['sha256'] != ARCHIVE_SHA:
            raise ValueError('archive differs from move53')
    return dict(runner=fact(Path(__file__)), bank=fact(BANK), leg=fact(LEG),
        sealed=source_files(SEALED), candidate=source_files(CANDIDATE),
        helpers=[fact(REPO / 'experiments' / name) for name in
                 ('ddm_mrs1_validate.py', 'ddm_mrs2_profile.py', 'ddm_mrs4_proof.py')],
        python=sys.executable, python_version=sys.version, platform=platform.platform(),
        compiler=subprocess.check_output(['cc', '--version'], text=True), compiler_path=shutil.which('cc'),
        frames=frames[:limit], seed=SEED, threads=4, device='cpu', limit=limit,
        scope=SCOPE if limit == 48 else SCOPE.replace('48 seeded stratified pairs', 'one bank pair (instrument smoke only)'),
        score_claim=False, authority=False, timing_clearance=False)


def run_sequence(root, inputs, limit, lock_fd):
    if shutil.disk_usage(ROOT).free < 2 * 1024**3:
        raise ValueError('storage preflight requires 2 GiB free SSD space')
    occupied = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.name.startswith('._'))
    if occupied + 4 * limit * (6104016 + 196608) > 2 * 1024**3:
        raise ValueError('conservative whole-arm retention quota would exceed 2 GiB; certify routing before launch')
    attempts = sorted(root.glob('sequence_[0-9]*'))
    attempt = root / f'sequence_{len(attempts) + 1:04d}'
    attempt.mkdir()
    atomic(attempt / 'INPUTS.json', inputs)
    outcomes = []
    for index, arm in enumerate(('A', 'B', 'A', 'B')):
        for ref in [inputs['runner'], inputs['bank'], inputs['leg'], *inputs['helpers'],
                    *inputs['sealed'], *inputs['candidate']]:
            verify(ref)
        run = attempt / f'{index + 1}_{arm}'
        run.mkdir()
        data = run / 'data'
        data.mkdir()
        runtime = SEALED
        if arm == 'B':
            runtime = run / 'runtime'
            runtime.mkdir()
            for ref in inputs['candidate']:
                source = Path(ref['path'])
                relative = source.relative_to(CANDIDATE)
                target = runtime / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied = fact(target)
                if (copied['bytes'], copied['sha256']) != (ref['bytes'], ref['sha256']):
                    raise ValueError('candidate snapshot copy differs')
        with zipfile.ZipFile(runtime / 'archive.zip') as archive:
            if archive.namelist() != ['p']:
                raise ValueError('unexpected archive member set')
            (data / 'p').write_bytes(archive.read('p'))
        file_list = run / 'files.txt'
        file_list.write_text('0.mkv\n')
        shim = run / 'shim'
        shim.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--arm', arm,
                   '--resume-from', str(run), '--runtime', str(runtime), '--limit', str(limit)]
        shell = ('#!/bin/sh\nset -eu\nif [ "$#" -gt 0 ] && [ "$1" = ' + shlex.quote(str(runtime / 'inflate.py')) + ' ]; then\n'
                 '  exec ' + shlex.join(command) + '\nfi\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
        for name in ('python', 'python3'):
            path = shim / name
            path.write_text(shell)
            path.chmod(0o755)
        env = os.environ.copy()
        for key in tuple(env):
            if key.startswith(('F26_', 'TC1_RECEIVER_', 'RLC1_', 'CPR1_', 'MRS5_')):
                env.pop(key)
        env.update(PATH=str(shim) + os.pathsep + env['PATH'], CC='cc', PYTHONDONTWRITEBYTECODE='1',
            RLC1_ADVISORY_CPU='1', RLC1_PROOF_BLAS_THREADS='4', F26_TOKEN_DECODER='python',
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
            VECLIB_MAXIMUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
        argv = ['bash', str(runtime / 'inflate.sh'), str(data), str(run / 'output'), str(file_list)]
        record = dict(arm=arm, sequence_index=index, argv=argv, shim=fact(shim / 'python'),
            shim_python3=fact(shim / 'python3'), before=uptime(), completed=False,
            score_claim=False, process_cold=True, cold_n600=False, scope=inputs['scope'])
        atomic(run / 'RUN.json', record)
        with (run / 'stdout.log').open('w') as out, (run / 'stderr.log').open('w') as error:
            execution, failure = execute_shell(argv, env, lock_fd, out, error)
        record.update(**execution, after=uptime(),
                      completed=failure is None and execution['returncode'] == 0,
                      stdout=fact(run / 'stdout.log'), stderr=fact(run / 'stderr.log'))
        atomic(run / 'RUN.json', record)
        if failure is not None:
            raise failure
        if execution['returncode'] != 0:
            raise ValueError('timing attempt failed; resume restarts a fresh complete A/B/A/B sequence')
        for ref in [inputs['runner'], inputs['bank'], inputs['leg'], *inputs['helpers'],
                    *inputs['sealed'], *inputs['candidate']]:
            verify(ref)
        worker_result = json.loads((run / 'WORKER_RESULT.json').read_text())
        if worker_result['pair_count'] != limit or not all(p['matches_bank'] for p in worker_result['pairs']):
            raise ValueError('worker did not retain all requested pair identities')
        record['worker_result'] = fact(run / 'WORKER_RESULT.json')
        record['pair_compute_seconds'] = worker_result['pair_compute_seconds']
        atomic(run / 'RUN.json', record)
        outcomes.append(record)
        atomic(attempt / 'CHECKPOINT.json', dict(completed_runs=len(outcomes), runs=outcomes,
            restart_policy='Never splice interrupted or old timing runs; a fresh sequence is required on resume'))
    a = [r['wall_seconds'] for r in outcomes if r['arm'] == 'A']
    b = [r['wall_seconds'] for r in outcomes if r['arm'] == 'B']
    ratio = statistics.median(b) / statistics.median(a)
    pure_a = [r['pair_compute_seconds'] for r in outcomes if r['arm'] == 'A']
    pure_b = [r['pair_compute_seconds'] for r in outcomes if r['arm'] == 'B']
    pairs = [b[i] / a[i] for i in range(2)]
    seconds = json.loads(LEG.read_text())['measured_t4_decode_seconds']
    result = dict(binding=inputs, runs=outcomes,
        ratio_median_b_over_median_a=ratio if limit == 48 else None,
        adjacent_ratios=pairs if limit == 48 else None,
        adjacent_ratio_half_range=(max(pairs) - min(pairs)) / 2 if limit == 48 else None,
        ratio_extreme_range=[min(b) / max(a), max(b) / min(a)] if limit == 48 else None,
        projected_t4_seconds=seconds * ratio if limit == 48 else None, source_t4_seconds=seconds,
        projection_assumption='Serial-Python scaling transfers; GPU stages assumed unchanged; host load approximately stationary',
        projection_is_timing_authority=False, no_statistical_confidence_interval=True,
        scope=inputs['scope'], complete=True, score_claim=False, authority=False,
        pair_compute_ratio_sensitivity=statistics.median(pure_b) / statistics.median(pure_a) if limit == 48 else None,
        pair_compute_seconds_a=pure_a, pair_compute_seconds_b=pure_b,
        charter_sample_complete=limit == 48, risk_limit_seconds=1260.0,
        projection_within_risk_limit=seconds * ratio <= 1260.0 if limit == 48 else None,
        instrument_smoke_only=limit != 48)
    atomic(attempt / 'RESULT.json', result)
    atomic(root / 'RESULT.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', required=True, type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--arm', choices=('A', 'B'))
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--limit', type=int, default=48)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    root = args.resume_from.resolve()
    if not root.is_relative_to(ROOT) or not root.relative_to(ROOT).parts[0].startswith('timing'):
        raise ValueError('timing paths must remain within charter-owned timing directory')
    if args.limit not in (1, 48):
        raise ValueError('only one-pair smoke or exact charter n48 allowed')
    root.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if args.arm is None or args.runtime is None:
            raise ValueError('worker requires exact arm/runtime')
        worker(root, args.arm, args.runtime.resolve(), args.limit)
        return
    lock = (root / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inputs = binding(args.limit)
    prior = root / 'INPUTS.json'
    if prior.exists() and json.loads(prior.read_text()) != inputs:
        raise ValueError('frozen sources changed; choose a new timing directory')
    atomic(prior, inputs)
    if args.prepare_only:
        print(json.dumps(dict(preflight='READY', input_receipt=str(prior), launches=0)))
        return
    if (root / 'RESULT.json').exists():
        result = json.loads((root / 'RESULT.json').read_text())
        if result['binding'] != inputs:
            raise ValueError('completed timing source binding differs')
        for run in result['runs']:
            verify(run['worker_result'])
            verify(run['stdout'])
            verify(run['stderr'])
            worker_result = json.loads(Path(run['worker_result']['path']).read_text())
            instrumentation = worker_result['instruction']
            for ref in [*instrumentation['native_libraries'].values(),
                        *instrumentation['extracted'], *instrumentation['original_extractions']]:
                verify(ref)
            for pair in worker_result['pairs']:
                for kind in ('token', 'raw'):
                    verify(pair[kind]['retained'])
        print(json.dumps(dict(status='complete_retained', result=str(root / 'RESULT.json'))))
        return
    run_sequence(root, inputs, args.limit, lock.fileno())


if __name__ == '__main__':
    main()
