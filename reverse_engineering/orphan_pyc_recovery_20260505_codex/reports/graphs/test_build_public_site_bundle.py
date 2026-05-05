"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``27:21: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``test_build_public_site_bundle.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'reports/graphs/test_build_public_site_bundle.py'
__recovery_spec__ = 'test_build_public_site_bundle.recovery_spec.json'
__recovery_ast_error__ = '27:21: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: test_build_public_site_bundle.cpython-312.pyc (Python 3.12)

from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from reports.graphs.build_public_site_bundle import build_public_site_bundle, sanitize_text

class PublicSiteBundleTests(unittest.TestCase):
    
    def test_sanitize_text_redacts_private_surfaces(self = None):
        text = 'path=/Users/adpena/Projects/pact/reports/raw/x tmp=/private/tmp/pact-mine/x ssh=ssh4.vast.ai:25850 url=https://lightning.ai/adpena/comma-lab/studios/foo/app?x=1 modal=ap-AbCdEf1234567890 call=fc-ABCDEF1234567890ABCDEF CLOUDFLARE_API_TOKEN=secret-token'
        (sanitized, records) = sanitize_text(text)
        self.assertNotIn('/Users/adpena', sanitized)
        self.assertNotIn('/private/tmp', sanitized)
        self.assertNotIn('ssh4.vast.ai', sanitized)
        self.assertNotIn('lightning.ai/adpena/comma-lab/studios', sanitized)
        self.assertNotIn('ap-AbCdEf1234567890', sanitized)
        self.assertNotIn('fc-ABCDEF1234567890ABCDEF', sanitized)
        self.assertIn('${LOCAL_PATH_REDACTED}', sanitized)
        self.assertIn('${VAST_SSH_REDACTED}', sanitized)
        self.assertIn('${LIGHTNING_PRIVATE_URL_REDACTED}', sanitized)
        self.assertIn('${MODAL_ID_REDACTED}', sanitized)
        self.assertIn('CLOUDFLARE_API_TOKEN=${SECRET_REDACTED}', sanitized)
        sum((lambda .0: pass# WARNING: Decompyle incomplete
)(records()), 6)

    
    def test_build_public_site_bundle_redacts_and_preserves_json(self = None):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp)
        source = root / 'site'
        output = root / 'public_site'
        source.mkdir()
        (source / 'index.html').write_text('<h1>Apogee</h1>\n', encoding = 'utf-8')
        (source / 'data.json').write_text(json.dumps({
            'path': '/Users/adpena/Projects/pact/private' }) + '\n', encoding = 'utf-8')
        (source / 'clip.bin').write_bytes(b'\x00\x01\x02')
        manifest = build_public_site_bundle(source, output, max_asset_bytes = 1024, strict_hygiene = False)
        self.assertTrue((output / 'index.html').is_file())
        self.assertEqual((output / 'clip.bin').read_bytes(), b'\x00\x01\x02')
        data = json.loads((output / 'data.json').read_text(encoding = 'utf-8'))
        self.assertEqual(data['path'], '${LOCAL_PATH_REDACTED}')
        self.assertEqual(manifest['redaction_count'], 1)
        self.assertTrue((output / 'public_site_manifest.json').is_file())
        None(None, None)
        return None
        with None:
            if not None:
                pass

    
    def test_build_public_site_bundle_rejects_large_assets(self = None):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp)
        source = root / 'site'
        output = root / 'public_site'
        source.mkdir()
        (source / 'large.mp4').write_bytes(b'xxxxxxxx')
        self.assertRaises(RuntimeError)
        build_public_site_bundle(source, output, max_asset_bytes = 4, oversized_policy = 'fail', strict_hygiene = False)
        None(None, None)
        None(None, None)
        return None
        with None:
            if not None:
                pass
        continue
        with None:
            if not None:
                pass

    
    def test_build_public_site_bundle_omits_large_assets_by_default(self = None):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp)
        source = root / 'site'
        output = root / 'public_site'
        source.mkdir()
        (source / 'large.gif').write_bytes(b'xxxxxxxx')
        manifest = build_public_site_bundle(source, output, max_asset_bytes = 4, strict_hygiene = False)
        self.assertFalse((output / 'large.gif').exists())
        self.assertEqual(manifest['omitted_oversized_assets'], [
            {
                'path': 'large.gif',
                'bytes': 8 }])
        None(None, None)
        return None
        with None:
            if not None:
                pass


if __name__ == '__main__':
    unittest.main()
    return None

"""
