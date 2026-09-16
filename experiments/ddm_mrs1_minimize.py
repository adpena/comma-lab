"""Reduce the frozen Python receiver using retained real-decode coverage."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import numpy as np

ROOT = Path('/Volumes/APDataStore/pact/ddm_mrs1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview', action='store_true', help='Inspection draft only, confined to preview.py in the charter store')
    args = parser.parse_args()
    evidence = json.loads(args.coverage.read_text())
    completed = evidence['completed_traced_pairs']
    planned = sorted(np.random.default_rng(20260916).choice(600, 32, replace=False).tolist())
    if (evidence.get('seed') != 20260916 or not evidence.get('trace_enabled')
            or evidence.get('planned_pairs') != planned or completed != sorted(set(completed))
            or any(pair not in planned for pair in completed)):
        raise ValueError('coverage is not a unique subset of the recorded seeded sample')
    inputs = json.loads(args.coverage.with_name('INPUTS.json').read_text())
    if inputs['archive']['sha256'] != 'aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957':
        raise ValueError('coverage belongs to another archive')
    source = Path(evidence['receiver']['path'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != evidence['receiver']['sha256']:
        raise ValueError('coverage source changed')
    if args.preview and args.output.resolve() != ROOT / 'preview.py':
        raise ValueError('preview output must be the explicitly unqualified inspection draft')
    if not args.preview and len(evidence['completed_traced_pairs']) < 24:
        raise ValueError('at least 24 actually traced seeded pairs are required')
    if not args.preview:
        identity = json.loads(args.coverage.with_name('IDENTITY.json').read_text())
        if identity['raw']['sha256'] != '8a14f55a6a8b141501836f511f4222dde75dca21714d923ad865b5f4757ef66b':
            raise ValueError('completed receiver trace lacks full host raw identity')
        if identity['binding'] != inputs or inputs['receiver'] != evidence['receiver']:
            raise ValueError('trace completion and source bindings disagree')
    hits = set(evidence['lines'])
    cold = json.loads((ROOT / 'start_trace_bound/COMPLETE.json').read_text())
    if cold['binding']['receiver'] != evidence['receiver'] or cold['binding']['archive'] != inputs['archive'] or cold['completed_pairs'] != [0]:
        raise ValueError('cold-start coverage belongs to another source or archive')
    if hashlib.sha256(Path(cold['token']['path']).read_bytes()).hexdigest() != cold['token']['sha256']:
        raise ValueError('cold-start token receipt changed')
    hits.update(cold['lines'])
    tree = ast.parse(source.read_text())
    mandatory = {'ArithmeticDecoder', 'CausalGeometry', 'parse_geometry_config', 'preceding_window',
                 'TokenDecoder', 'read_models', 'write_video'}
    deleted_functions = set()

    class Prune(ast.NodeTransformer):
        def __init__(self):
            self.depth = 0
            self.protected = 0

        def live(self, node):
            return any(line in hits for line in range(node.lineno, node.end_lineno + 1))

        def visit_FunctionDef(self, node):
            if node.name == 'main':
                return None
            keep = self.protected or node.name in mandatory
            if not keep and not any(self.live(n) for n in node.body):
                deleted_functions.add(node.name)
                return None
            self.depth += 1
            self.protected += bool(keep)
            node = self.generic_visit(node)
            self.protected -= bool(keep)
            self.depth -= 1
            node.body = node.body or [ast.Pass()]
            return node

        def visit_ClassDef(self, node):
            keep = node.name in mandatory
            self.protected += keep
            node = self.generic_visit(node)
            self.protected -= keep
            node.body = node.body or [ast.Pass()]
            return node

        def visit_If(self, node):
            if self.protected or not self.depth:
                return self.generic_visit(node)
            body_live = any(self.live(n) for n in node.body)
            other_live = any(self.live(n) for n in node.orelse)
            # Retain explicit rejection guards; they do not add alternative codecs.
            guard = all(isinstance(n, (ast.Raise, ast.Assert)) for n in node.body)
            if body_live or guard:
                node.orelse = node.orelse if other_live else []
                node = self.generic_visit(node)
                node.body = node.body or [ast.Pass()]
                return node
            if other_live:
                return [self.visit(n) for n in node.orelse]
            return None

        def generic_visit(self, node):
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                return node
            if isinstance(node, ast.stmt) and self.depth and not self.protected:
                if not self.live(node) and not isinstance(node, (ast.Raise, ast.Assert)):
                    return None
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name) and node.value.func.id == 'print':
                    return None
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'started' for target in node.targets):
                return None
            if isinstance(node, ast.If) and '__name__' in ast.unparse(node.test):
                return None
            return super().generic_visit(node)

    tree = Prune().visit(tree)
    class DropUnusedFactoryEntries(ast.NodeTransformer):
        def visit_Dict(self, node):
            kept = [(key, value) for key, value in zip(node.keys, node.values)
                    if not ({n.id for n in ast.walk(value) if isinstance(n, ast.Name)} & deleted_functions)]
            node.keys = [key for key, _ in kept]
            node.values = [value for _, value in kept]
            return self.generic_visit(node)
    tree = DropUnusedFactoryEntries().visit(tree)
    direct = ast.parse(Path('experiments/ddm_mrs1_direct_carrier.inc').read_text() + '\n' +
                       Path('experiments/ddm_mrs1_direct_prior.inc').read_text())
    replacements = {node.name for node in direct.body if isinstance(node, ast.FunctionDef)}
    tree.body = [node for node in tree.body if not isinstance(node, ast.FunctionDef) or node.name not in replacements]
    tree.body += direct.body
    ast.fix_missing_locations(tree)
    text = ast.unparse(tree)
    text = text.replace('        prior_blob = prior_format_materialize_ihs1(parts.hpac_blob, render_namespace)\n', '')
    text = text.replace('self.model = render_load_hpac(prior_blob, device)', 'self.model = render_load_hpac(parts.hpac_blob, device)')
    text = text.replace('    model_module = prior_namespace\n', '')
    substitutions = {
        'def weight_mixer_walk(descriptors, weights, *, source=None, payload=None, observe=False):':
            'def weight_mixer_walk(descriptors, weights, *, payload):',
        '    encoder = model_bits_RangeEncoder() if source is not None and (not observe) else None\n': '',
        '    decoder = model_bits_RangeDecoder(payload) if payload is not None else None':
            '    decoder = model_bits_RangeDecoder(payload)',
        '    state, groups, events, truth = (weight_mixer_Predictors(), [], [], [])':
            '    state, groups = weight_mixer_Predictors(), []',
        '    return (encoder.finish() if encoder is not None else None, groups, events, truth)': '    return groups',
        '    _, groups, _, _ = weight_mixer_walk(': '    groups = weight_mixer_walk(',
        'def prior_mixer_walk(counts, depths, family, weights, learning_shift, *, source_rows=None, payload=None, trace=False):':
            'def prior_mixer_walk(counts, depths, family, weights, learning_shift, *, payload):',
        '    encoder = adaptive_RangeEncoder() if source_rows is not None else None\n': '',
        '    decoder = adaptive_RangeDecoder(payload) if payload is not None else None':
            '    decoder = adaptive_RangeDecoder(payload)',
        '    result, events = ([], [])': '    result = []',
        '    return (encoder.finish() if encoder is not None else None, result, events)': '    return result',
        '    _, rows, _ = prior_mixer_walk(': '    rows = prior_mixer_walk(',
        'refusing to encode': 'refusing to decode',
    }
    for old, new in substitutions.items():
        if old not in text:
            raise ValueError(f'frozen source specialization anchor is missing: {old}')
        text = text.replace(old, new)
    text = text.replace("render_namespace = SimpleNamespace(**{name.removeprefix('render_'): value for name, value in list(globals().items()) if name.startswith('render_')}, IntegerHPAC=prior_IntegerHPAC)",
        'render_namespace = SimpleNamespace(IntegerHPAC=prior_IntegerHPAC, N=render_N, NUM_CLASSES=render_NUM_CLASSES, HPAC_PATCH=render_HPAC_PATCH, HPAC_DELTA=render_HPAC_DELTA, HPAC_CHANNELS=render_HPAC_CHANNELS, HPAC_FILM_DIM=render_HPAC_FILM_DIM, CARRIER_DIM=render_CARRIER_DIM)')
    tree = ast.parse(text)
    aliases = {}
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                natural = {'numpy': 'np'}.get(name.name, name.name)
                aliases[name.asname or name.name] = natural
                imports[natural] = ast.Import(names=[ast.alias(name.name, natural if natural != name.name else None)])
        elif isinstance(node, ast.ImportFrom):
            for name in node.names:
                natural = 'F' if name.name == 'functional' else name.name
                aliases[name.asname or name.name] = natural
                imports[natural] = ast.ImportFrom(module=node.module, names=[ast.alias(name.name, natural if natural != name.name else None)], level=0)

    rename = {'Fx2ModelAxisMixer': 'ContextOddsMixer', 'Ma1WithinMissCorrector': 'MissCorrector',
        'IHS2Layout': 'PriorLayout', 'IHS2Error': 'PriorFormatError', 'MP2SemanticFormatError': 'WeightFormatError',
        'Rc2CodecError': 'ModelCodecError', 'fx2_': '', 'tc1_weights': 'mixer_parameters',
        'materialize_ihs1': 'restore_prior', 'walk_sm3r': 'walk_renderer_rows',
        'decode_rx1_models': 'decode_model_sections', 'ck2_uninterleave_planes': 'interleave_bytes',
        'RC1_RESERVED_HPAC_ADAPTIVE': 'PRIOR_ADAPTIVE', 'RC1_RESERVED_SEMANTIC_ADAPTIVE': 'RENDERER_ADAPTIVE',
        'RR5_RESERVED_ARITH_BASIS': 'ARITHMETIC_BASIS', 'DX2_RESERVED_CABAC_COEFFICIENTS': 'ARITHMETIC_COEFFICIENTS',
        'CK2_RESERVED_SEMANTIC_PLANE2': 'RENDERER_PLANES', 'RX1_MODEL_HEADER': 'MODEL_HEADER',
        'RX1_CODEC_XZ': 'XZ_CODEC', 'HPAC': 'Prior', 'hpac': 'prior', 'materialize_cpr1': 'restore_carrier',
        'RLC1': 'Geometry', 'TC1': 'Shared', 'RC3': 'Prior', 'SM1': 'Renderer',
        'F26': 'Decoder', 'RR4': 'Counts', 'IHS2': 'Prior', 'FX1': 'Odds'}
    rename.update({'_cpr1_sparse_cache': '_sparse_cache', 'rx1': 'model_sections',
                   'SM3R': 'RENDERER_ROWS'})

    def renamed(value):
        value = aliases.get(value, value)
        for old, new in rename.items():
            value = value.replace(old, new)
        return value

    class Rename(ast.NodeTransformer):
        def visit_Import(self, node):
            return None
        visit_ImportFrom = visit_Import

        def visit_Name(self, node):
            node.id = renamed(node.id)
            return node

        def visit_Attribute(self, node):
            node.attr = renamed(node.attr)
            return self.generic_visit(node)

        def visit_arg(self, node):
            node.arg = renamed(node.arg)
            return self.generic_visit(node)

        def visit_keyword(self, node):
            if node.arg:
                node.arg = renamed(node.arg)
            return self.generic_visit(node)

        def visit_FunctionDef(self, node):
            node.name = renamed(node.name)
            return self.generic_visit(node)

        visit_ClassDef = visit_FunctionDef

        def visit_Constant(self, node):
            if isinstance(node.value, bytes):
                return ast.Call(func=ast.Attribute(value=ast.Name(id='bytes', ctx=ast.Load()), attr='fromhex', ctx=ast.Load()), args=[ast.Constant(node.value.hex())], keywords=[])
            if isinstance(node.value, str):
                node.value = renamed(node.value)
            return node

    tree = Rename().visit(tree)
    tree.body = list(imports.values()) + tree.body
    tree.body += ast.parse(Path('experiments/ddm_mrs1_public_main.inc').read_text()).body

    def defined(node):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            return {node.name}
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return {name.asname or name.name for name in node.names}
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            return {n.id for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                    for n in ast.walk(target) if isinstance(n, ast.Name)}
        return set()

    def loaded(node):
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}

    needed = {'main', 'TokenDecoder', 'read_models', 'write_video'}
    while True:
        before = needed.copy()
        for node in tree.body:
            if defined(node) & needed or isinstance(node, ast.Expr) and loaded(node) & needed:
                needed |= loaded(node)
        if before == needed:
            break
    tree.body = [node for node in tree.body if defined(node) & needed or
                 isinstance(node, ast.Expr) and loaded(node) & needed or
                 isinstance(node, ast.If) and '__name__' in ast.unparse(node.test)]
    ast.fix_missing_locations(tree)
    text = ast.unparse(tree) + '\n'
    # Drop inert progress conditions left after removal of the original prints.
    text = text.replace('        if end % 50 == 0 or end == render_N:\n            pass\n', '')
    text = text.replace('        if end % 64 == 0 or end == render_N:\n            pass\n', '')
    text = text.replace("        if tokens[:4] == bytes.fromhex('524c4331'):\n            pass\n", '')
    text = text.replace('import brotli\n', "try:\n    import brotli\nexcept ImportError:\n    raise SystemExit('Brotli is required: install brotli in the Python environment.')\n")
    forbidden = [line for line in text.splitlines() if re.search(r'ddm_|fx1|fx2|rr4|ihs2|tc1|rc3|rlc1|f26|sm1|cpr1|rx1|sm3r|\benviron\b|checkpoint|resume|advisory', line, re.I)]
    if forbidden:
        raise ValueError(f'internal identifier or forbidden runtime mechanism remains: {forbidden}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('"""Decode class planes with a learned prior, then render RGB and a pose carrier."""\n' + text)
    print(json.dumps(dict(path=str(args.output), lines=len(args.output.read_text().splitlines()),
                         covered_pairs=len(evidence['completed_traced_pairs']), qualified=not args.preview)))


if __name__ == '__main__':
    main()
