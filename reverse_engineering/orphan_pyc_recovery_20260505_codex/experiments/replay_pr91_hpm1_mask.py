# pyc-recovery: STUB unreconstructible -- see .recovery_spec.json for dis() ground-truth
# pycdc could not produce parseable output; raw decompiled text preserved in _PYCDC_PARTIAL_OUTPUT below.
"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``48:21: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``replay_pr91_hpm1_mask.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/replay_pr91_hpm1_mask.py'
__recovery_spec__ = 'replay_pr91_hpm1_mask.recovery_spec.json'
__recovery_ast_error__ = '48:21: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: replay_pr91_hpm1_mask.cpython-312.pyc (Python 3.12)

'''Replay PR91 HPM1 mask tokens locally and fail closed on parity gaps.'''
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from tac.pr91_hpm1_codec import DEFAULT_HPAC_PROBABILITY_VARIANT, DEFAULT_PR85_QMA9_TOKEN_SOURCE, DEFAULT_PR85_STBM_ADJUDICATED_JSON, DEFAULT_PR85_STBM_ARCHIVE, DEFAULT_PR91_ARCHIVE, extract_pr91_hpm1_payload, plan_pr91_hpm1_pr85_stbm_fusion, prototype_reencode_hpm1_from_raw_tokens, prototype_reencode_hpm1_residual_from_raw_tokens, run_pr91_hpm1_first_symbol_state_probe, run_pr91_hpm1_context_window_probe, run_pr91_hpm1_probability_variant_matrix, run_pr91_hpm1_preflight

def build_arg_parser():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--archive', type = Path, default = DEFAULT_PR91_ARCHIVE)
    parser.add_argument('--max-frames', type = int, default = 1, help = 'Local decode cap. Use --full-decode to omit the cap.')
    parser.add_argument('--full-decode', action = 'store_true', help = 'Attempt full 600-frame HPM1 decode. Local only; no scorer loads.')
    parser.add_argument('--attempt-reencode', action = 'store_true', help = 'If decode passes, re-encode decoded tokens. Full byte parity is meaningful only with --full-decode.')
    parser.add_argument('--raw-token-bin', type = Path, default = None, help = 'Optional decoded uint8 NHW token file for the local-only re-encode prototype.')
    parser.add_argument('--raw-token-shape', default = '600,512,384', help = 'Shape for --raw-token-bin before layout normalization. The default matches the PR85 QMA9 storage-order token source.')
    parser.add_argument('--raw-token-layout', default = 'qma9_storage_wh_to_render_hw', choices = ('qma9_storage_wh_to_render_hw', 'nhw_render_order'), help = 'Normalize --raw-token-bin into render-order N,H,W before prototype encoding.')
    parser.add_argument('--prototype-max-frames', type = int, default = None, help = 'Frame cap for --raw-token-bin prototype encoding.')
    parser.add_argument('--residual-symbols', action = 'store_true', help = 'With --raw-token-bin, encode mod-5 residual symbols while conditioning on raw previous-frame context. Local-only; no score claim or dispatch.')
    parser.add_argument('--first-symbol-state-probe', action = 'store_true', help = 'Write a local-only trace of the first submitted HPM1 symbols and probability rows.')
    parser.add_argument('--context-window-probe', action = 'store_true', help = 'Replay bounded PR91 HPM1 symbol windows under decoded and teacher-forced reference contexts.')
    parser.add_argument('--probability-variant-matrix', action = 'store_true', help = 'Probe all requested HPAC probability contracts and fail closed unless full byte parity is proven.')
    parser.add_argument('--symbol-count', type = int, default = 16, help = 'Number of prefix symbols to trace with --first-symbol-state-probe.')
    parser.add_argument('--symbol-offset', type = int, default = 0, help = 'Global decoded-symbol offset for --first-symbol-state-probe window tracing.')
    parser.add_argument('--symbol-windows', default = '33:8,5948:8', help = 'Comma-separated start:count windows for --context-window-probe. Defaults to the first context divergence and entropy-failure window.')
    parser.add_argument('--context-modes', default = 'decoded_context,reference_context', help = 'Comma-separated context modes for --context-window-probe.')
    parser.add_argument('--prob-eps-values', default = '1e-7', help = 'Comma-separated probability clip eps values for --context-window-probe.')
    parser.add_argument('--reference-tokens', type = Path, default = DEFAULT_PR85_QMA9_TOKEN_SOURCE, help = 'Optional PR85/QMA9 decoded uint8 NHW token source for prefix comparison.')
    parser.add_argument('--reference-layout', default = 'qma9_storage_wh_to_render_hw', choices = ('qma9_storage_wh_to_render_hw', 'legacy_assume_nhw'), help = 'How to interpret --reference-tokens before comparing to PR91 render-order symbols.')
    parser.add_argument('--probability-variants', default = None, help = 'Comma-separated HPAC probability variants. Defaults to the source contract for --first-symbol-state-probe and all registered variants for --probability-variant-matrix.')
    parser.add_argument('--fusion-plan', action = 'store_true', help = 'Plan the PR85+STBM -> PR91/HPM1 byte-faithful fusion gate without dispatch.')
    parser.add_argument('--pr85-stbm-archive', type = Path, default = DEFAULT_PR85_STBM_ARCHIVE)
    parser.add_argument('--pr85-stbm-adjudicated-json', type = Path, default = DEFAULT_PR85_STBM_ADJUDICATED_JSON)
    parser.add_argument('--skip-fusion-prefix-probe', action = 'store_true', help = 'Skip the local HPM1 prefix decode probe inside --fusion-plan.')
    parser.add_argument('--json-out', type = Path, default = None)
    return parser


def _load_raw_tokens(path = None, shape_text = None, layout = None):
    import numpy as np
    shape = (lambda .0: pass# WARNING: Decompyle incomplete
)(shape_text.split(',')())
    if len(shape) != 3:
        raise SystemExit('--raw-token-shape must have three dimensions')
    raw = path.read_bytes()
    expected = shape[0] * shape[1] * shape[2]
    if len(raw) != expected:
        raise SystemExit(f'''raw token size mismatch: got {len(raw)} expected {expected}''')
    arr = np.frombuffer(raw, dtype = np.uint8).reshape(shape)
    if layout == 'qma9_storage_wh_to_render_hw':
        arr = np.transpose(arr, (0, 2, 1)).copy()
    elif layout == 'nhw_render_order':
        arr = np.ascontiguousarray(arr)
    else:
        raise SystemExit(f'''unsupported raw token layout: {layout}''')
    if arr.size:
        if int(arr.min()) < 0 or int(arr.max()) > 4:
            raise SystemExit('raw token class value out of range 0..4')
    return arr


def _parse_probability_variants(text = None, *, default_source):
    pass
# WARNING: Decompyle incomplete


def _parse_symbol_windows(text = None):
    windows = []
    for item in text.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise SystemExit(f'''bad --symbol-windows item {item!r}; expected start:count''')
        (start_text, count_text) = item.split(':', 1)
        windows.append((int(start_text), int(count_text)))
    if not windows:
        raise SystemExit('--symbol-windows must specify at least one start:count pair')
    return tuple(windows)


def _parse_csv(text = None):
    values = (lambda .0: pass# WARNING: Decompyle incomplete
)(text.split(',')())
    if not values:
        raise SystemExit('comma-separated value must not be empty')
    return values


def _parse_prob_eps_values(text = None):
    return (lambda .0: pass# WARNING: Decompyle incomplete
)(_parse_csv(text)())


def main(argv = None):
    args = build_arg_parser().parse_args(argv)
    if args.fusion_plan:
        report = plan_pr91_hpm1_pr85_stbm_fusion(pr85_stbm_archive = args.pr85_stbm_archive, pr91_archive = args.archive, pr85_stbm_adjudicated_json = args.pr85_stbm_adjudicated_json, include_hpm1_prefix_probe = not (args.skip_fusion_prefix_probe))
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    raise SystemExit(main())

"""
