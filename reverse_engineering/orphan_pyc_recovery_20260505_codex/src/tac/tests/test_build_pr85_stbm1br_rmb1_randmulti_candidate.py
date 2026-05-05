# pyc-recovery: STUB unreconstructible -- see .recovery_spec.json for dis() ground-truth
# pycdc could not produce parseable output; raw decompiled text preserved in _PYCDC_PARTIAL_OUTPUT below.
"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``44:13: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``test_build_pr85_stbm1br_rmb1_randmulti_candidate.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/tests/test_build_pr85_stbm1br_rmb1_randmulti_candidate.py'
__recovery_spec__ = 'test_build_pr85_stbm1br_rmb1_randmulti_candidate.recovery_spec.json'
__recovery_ast_error__ = '44:13: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: test_build_pr85_stbm1br_rmb1_randmulti_candidate.cpython-312.pyc (Python 3.12)

from __future__ import annotations
import importlib.util as importlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any
import pytest
brotli = pytest.importorskip('brotli')
from tac.pr85_bundle import FIXED_V5_LENGTHS, PR85_HEADERLESS_RANDMULTI_SPECS, pack_pr85_bundle, parse_pr85_bundle
from tac.stbm1br_mask_codec import STBM1BR_MAGIC
REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / 'experiments' / 'build_pr85_stbm1br_rmb1_randmulti_candidate.py'

def _load_module():
    spec = importlib.util.spec_from_file_location('build_pr85_stbm1br_rmb1_test', SCRIPT)
# WARNING: Decompyle incomplete


def _zip_info():
    info = zipfile.ZipInfo('x', (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 27525120
    info.create_system = 3
    return info


def _row(counts = None):
    out = bytearray()
    out.append(len(counts))
    last = -1
    for idx, _value in counts:
        delta = idx - last - 1
        last = idx
        byte = delta & 127
        delta >>= 7
        out.append(byte | 128 if delta else byte)
        if not delta:
            continue
    continue
    (lambda .0: pass# WARNING: Decompyle incomplete
)(counts())
    return bytes(out)


def _headerless_rows():
    rows = [
        _row([
            (0, 5),
            (3, 7)])]
    range(sum((lambda .0: pass# WARNING: Decompyle incomplete
)(PR85_HEADERLESS_RANDMULTI_SPECS()) - 1)())
    return b''.join(rows)


def _rmb1_from_rows(raw = None):
    cursor = 0
    mask = bytearray()
    values = bytearray()
    if cursor < len(raw):
        count = raw[cursor]
        cursor += 1
        if count == 255:
            count = int.from_bytes(raw[cursor:cursor + 2], 'little')
            cursor += 2
        row_mask = bytearray(75)
        idx = -1
        for _ in range(count):
            delta = 0
            shift = 0
            byte = raw[cursor]
            cursor += 1
            delta |= (byte & 127) << shift
            if byte < 128:
                pass
            else:
                shift += 7
            idx += delta + 1
        values.extend(raw[cursor:cursor + count])
        cursor += count = None
        mask.extend(row_mask)
        if cursor < len(raw):
            continue
    mask_br = brotli.compress(bytes(mask), quality = 5)
    vals_br = brotli.compress(bytes(values), quality = 5)
    return b'RMB1' + len(mask_br).to_bytes(2, 'little') + mask_br + vals_br


def _write_archive(path = None, *, mask, randmulti):
    segments = {
        'mask': mask,
        'model': brotli.compress(b'QH0mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm', quality = 5),
        'pose': brotli.compress(b'P1D1pppppppppppppppppppppppppppppppp', quality = 5),
        'post': brotli.compress(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', quality = 5),
        'shift': brotli.compress(b'SD4\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', quality = 5),
        'frac': brotli.compress(b'FH1\x04\x04\x04\x04\x04\x04\x04\x04\x04\x04\x04\x04', quality = 5),
        'frac2': brotli.compress(b'FH2\x04\x04\x04\x04\x04\x04\x04\x04\x04\x04\x04\x04', quality = 5),
        'frac3': brotli.compress(b'FD3\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', quality = 5),
        'bias': b'B' * FIXED_V5_LENGTHS['bias'],
        'region': b'R' * FIXED_V5_LENGTHS['region'],
        'randmulti': randmulti }
    raw = pack_pr85_bundle(segments, header_mode = 'v5')
    path.parent.mkdir(parents = True, exist_ok = True)
    zf = zipfile.ZipFile(path, 'w')
    zf.writestr(_zip_info(), raw)
    None(None, None)
    return path
    with None:
        if not None:
            pass
    return path


def _write_runtime_support(root = None, *, support_x):
    root.mkdir(parents = True, exist_ok = True)
    (root / 'inflate_renderer.py').write_text("STBM1BR_MAGIC = b'STBM1BR\\0'\ndef _load_masks_from_stbm1br():\n    pass\n", encoding = 'utf-8')
    (root / 'apply_qzs3_postprocess.py').write_text('def _decode_rmb1_randmulti_payload():\n    pass\nif blob[:4] == b"RMB1":\n    pass\n', encoding = 'utf-8')
    x_probe = 'if [ -f "$ARCHIVE_DIR/x" ]; then echo x; fi\n' if support_x else ''
    (root / 'inflate.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n' + x_probe, encoding = 'utf-8')
    (root / 'unpack_renderer_payload.py').write_text('PAYLOAD_SHORT_BR = "x"\n' if support_x else 'PAYLOAD_SHORT_BR = "p"\n', encoding = 'utf-8')
    return root


def test_builder_replaces_only_randmulti_and_records_parity(tmp_path = None, monkeypatch = None):
    module = _load_module()
    rows = _headerless_rows()
    stbm_archive = _write_archive(tmp_path / 'stbm.zip', mask = STBM1BR_MAGIC + b'mask', randmulti = brotli.compress(rows, quality = 0))
    pr92_archive = _write_archive(tmp_path / 'pr92.zip', mask = b'QMA9mask', randmulti = _rmb1_from_rows(rows))
    monkeypatch.setattr(module, 'EXPECTED_STBM_SHA256', module._sha256_file(stbm_archive))
    monkeypatch.setattr(module, 'EXPECTED_STBM_BYTES', stbm_archive.stat().st_size)
    monkeypatch.setattr(module, 'EXPECTED_PR92_SHA256', module._sha256_file(pr92_archive))
    monkeypatch.setattr(module, 'ROBUST_CURRENT_DIR', _write_runtime_support(tmp_path / 'runtime'))
    summary = module.build_pr85_stbm1br_rmb1_randmulti_candidate(stbm_archive = stbm_archive, pr92_archive = pr92_archive, out_dir = tmp_path / 'out')
    manifest = json.loads((tmp_path / 'out' / summary['candidate_id'] / 'manifest.json').read_text())
# WARNING: Decompyle incomplete


def test_builder_fails_closed_when_rmb1_rows_differ(tmp_path = None, monkeypatch = None):
    module = _load_module()
    rows = _headerless_rows()
    stbm_archive = _write_archive(tmp_path / 'stbm.zip', mask = STBM1BR_MAGIC + b'mask', randmulti = brotli.compress(rows, quality = 5))
    pr92_archive = None(_rmb1_from_rows, mask = b'\x00', randmulti = None(sum * (lambda .0: pass# WARNING: Decompyle incomplete
)(PR85_HEADERLESS_RANDMULTI_SPECS())))
    monkeypatch.setattr(module, 'EXPECTED_STBM_SHA256', module._sha256_file(stbm_archive))
    monkeypatch.setattr(module, 'EXPECTED_STBM_BYTES', stbm_archive.stat().st_size)
    monkeypatch.setattr(module, 'EXPECTED_PR92_SHA256', module._sha256_file(pr92_archive))
    monkeypatch.setattr(module, 'ROBUST_CURRENT_DIR', _write_runtime_support(tmp_path / 'runtime'))
    pytest.raises(module.Rmb1CandidateBuildError, match = 'decoded randmulti rows differ')
    module.build_pr85_stbm1br_rmb1_randmulti_candidate(stbm_archive = stbm_archive, pr92_archive = pr92_archive, out_dir = tmp_path / 'out')
    None(None, None)
    return None
    with None:
        if not b'QMA9mask':
            pass
    tmp_path / 'pr92.zip'
    _write_archive


def test_builder_fails_closed_without_single_member_x_runtime_support(tmp_path = None, monkeypatch = None):
    module = _load_module()
    rows = _headerless_rows()
    stbm_archive = _write_archive(tmp_path / 'stbm.zip', mask = STBM1BR_MAGIC + b'mask', randmulti = brotli.compress(rows, quality = 0))
    pr92_archive = _write_archive(tmp_path / 'pr92.zip', mask = b'QMA9mask', randmulti = _rmb1_from_rows(rows))
    monkeypatch.setattr(module, 'EXPECTED_STBM_SHA256', module._sha256_file(stbm_archive))
    monkeypatch.setattr(module, 'EXPECTED_STBM_BYTES', stbm_archive.stat().st_size)
    monkeypatch.setattr(module, 'EXPECTED_PR92_SHA256', module._sha256_file(pr92_archive))
    monkeypatch.setattr(module, 'ROBUST_CURRENT_DIR', _write_runtime_support(tmp_path / 'runtime', support_x = False))
    pytest.raises(module.Rmb1CandidateBuildError, match = 'robust_current_runtime_support')
    module.build_pr85_stbm1br_rmb1_randmulti_candidate(stbm_archive = stbm_archive, pr92_archive = pr92_archive, out_dir = tmp_path / 'out')
    None(None, None)
    return None
    with None:
        if not None:
            pass


"""
