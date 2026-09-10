"""Real-source CPU writer parity and retained-stage controls for VR8."""
import ast
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import ddm_vr8_raw_restore as raw


def test_copy_retains_interrupted_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(raw.common, 'go', lambda: None)
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.write_bytes(b'abc')
    target.write_bytes(b'ab')
    with pytest.raises(ValueError, match='interrupted'):
        raw.copy_pinned(raw.vr7.fact(source), target)
    assert target.read_bytes() == b'ab'


def test_copy_resume_verifies_without_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(raw.common, 'go', lambda: None)
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.write_bytes(b'abc')
    expected = raw.vr7.fact(source)
    raw.copy_pinned(expected, target)
    identity = target.stat().st_mtime_ns
    raw.copy_pinned(expected, target)
    assert target.stat().st_mtime_ns == identity


@pytest.mark.parametrize('relative', [
    'ddm_sj1_pass5_price/candidate_pass5/candidate_runtime',
    'ddm_sj1_multipass_token_predistortion/candidate/candidate_runtime',
    'ddm_sj1_multipass_token_predistortion/candidate_pass3/candidate_runtime',
])
def test_real_original_render_function_matches_pair_adapter(tmp_path, relative):
    torch = pytest.importorskip('torch')
    np = pytest.importorskip('numpy')
    path = Path('/Volumes/VertigoDataTier/pact') / relative / 'cpr1/inflate.py'
    if not path.is_file():
        pytest.skip('retained receiver unavailable')
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('normalized_basis', 'render_video')]
    assert len(functions) == 2
    # Execute the unchanged source functions with deliberately small synthetic geometry.
    namespace = {'torch': torch, 'np': np, 'math': math, 'time': time, 'Path': Path,
                     'functional': torch.nn.functional, 'N': 4, 'CAMERA_H': 7, 'CAMERA_W': 9,
                     'EVAL_H': 4, 'EVAL_W': 5, 'CARRIER_DIM': 2, 'CARRIER_AMPLITUDE': 9.0}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    renderer = SimpleNamespace(**namespace)

    class Semantic(torch.nn.Module):
        def forward(self, tokens, indices):
            # Global pair indices affect pixels, detecting accidental local reindexing.
            base = tokens[:, None].float().repeat(1, 3, 1, 1)
            return base * 11.3 + indices[:, None, None, None] * 7.1

    torch.manual_seed(17)
    device = torch.device('cpu')
    state = {'semantic': Semantic(), 'basis': torch.randn(2, 3, 2, 3),
                 'coefficients': torch.randn(4, 2), 'tokens': torch.randint(0, 5, (4, 4, 5)), 'device': device}
    original = tmp_path / 'original.raw'
    renderer.render_video(state['semantic'], state['basis'], state['coefficients'],
                          state['tokens'], original, device)
    pair = raw.pair_renderer(renderer, None, state, None)
    actual = tmp_path / 'pairs.raw'
    actual.write_bytes(b''.join(pair(i) for i in range(4)))
    assert actual.read_bytes() == original.read_bytes()


@pytest.mark.parametrize('mode_index', [0, 1])
def test_selector_applies_to_slave_only(tmp_path, mode_index):
    torch = pytest.importorskip('torch')
    np = pytest.importorskip('numpy')
    renderer = SimpleNamespace(CAMERA_H=2, CAMERA_W=3, CARRIER_DIM=1, CARRIER_AMPLITUDE=1,
                               normalized_basis=lambda x: x)
    class Semantic(torch.nn.Module):
        def forward(self, tokens, indices):
            return torch.zeros((1, 3, 2, 3)) + 31
    state = {'device': torch.device('cpu'), 'semantic': Semantic(), 'basis': torch.zeros((1, 3, 2, 3)),
                 'coefficients': torch.ones((1, 1)), 'tokens': torch.zeros((1, 2, 3))}
    calls = []
    def mode(values, delta):
        calls.append(values.copy())
        return (values.astype(np.int16) + delta).astype(np.uint8)
    pair = raw.pair_renderer(renderer, SimpleNamespace(apply_pixel_mode=mode), state, ([1, 1], [mode_index]))
    data = pair(0)
    retained = tmp_path / 'selected.raw'
    retained.write_bytes(data)
    assert data[:18] == bytes([129]) * 18
    assert data[18:] == bytes([31]) * 18
    assert len(calls) == 1


def test_pair_suffix_refuses_without_rendering(tmp_path, monkeypatch):
    monkeypatch.setattr(raw.common, 'go', lambda: None)
    output = tmp_path / '0.raw'
    output.write_bytes(b'x')
    root = tmp_path / 'receipts'
    root.mkdir()
    with pytest.raises(ValueError, match='suffix retained'):
        raw.write_pairs({'expected_bytes': 2400}, output, root, {},
                        lambda i: pytest.fail('must not render'), 4)
    assert output.read_bytes() == b'x'


def test_complete_pair_writer_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(raw.common, 'go', lambda: None)
    monkeypatch.setattr(raw.common, 'reserve', lambda *args: None)
    monkeypatch.setattr(raw.vr7, 'verify', lambda source: None)
    output, root = tmp_path / '0.raw', tmp_path / 'receipts'
    root.mkdir()
    source = {'expected_bytes': 2400, 'expected_sha256': hashlib.sha256(b'abcd' * 600).hexdigest()}
    binding = {'consumer_store': 'test'}
    assert raw.write_pairs(source, output, root, binding, lambda i: b'abcd', 4) == 0
    assert raw.write_pairs(source, output, root, binding,
                           lambda i: pytest.fail('completed payload re-rendered'), 4) == 0
    assert json.loads((root / 'RESTORE_CERTIFICATE.json').read_text())['complete']
