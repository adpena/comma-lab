# ddm_fm2 fmtools Patch Apply Instructions

Base checkout:
- path: `/Users/adpena/Projects/fmtools`
- expected commit: `c9e755539da22df8aee6c5c22fa6653253456a4f`
- expected status before apply: clean `main`

Patch:
- `0001-fmtools-full-sdk-surfaces.patch`
- sha256: `9804876bec64c21b1de70470bedfd276d12e3c0ab6cace3ef3c24ecbbd6340ee`
- line count: 804

Apply:

```bash
cd /Users/adpena/Projects/fmtools
git apply /Users/adpena/Projects/pact/.omx/research/ddm_fm2_20260806/fmtools_patches/0001-fmtools-full-sdk-surfaces.patch
```

Post-apply verification used in the writable SSD clone:

```bash
PYTHONPATH=/Volumes/VertigoDataTier/pact/ddm_fm2_20260806/fmtools_patch_verify /Users/adpena/Projects/fmtools/.venv/bin/python -m pytest tests
/Users/adpena/Projects/fmtools/.venv/bin/ruff check fmtools/capabilities.py fmtools/session.py fmtools/protocols.py fmtools/backends/apple_sdk.py fmtools/backends/ffi.py fmtools/decorators.py fmtools/__init__.py tests/test_full_sdk_surfaces.py tests/test_packaging_metadata.py
```

Observed:
- `git apply --check` against `/Users/adpena/Projects/fmtools`: clean.
- Full fmtools tests in SSD clone: 652 passed, 12 skipped.
- Ruff on changed fmtools files: all checks passed.

Boundary:
- This Codex sandbox could not write `/Users/adpena/Projects/fmtools`; patch series is the deliverable instead of an in-place branch.
