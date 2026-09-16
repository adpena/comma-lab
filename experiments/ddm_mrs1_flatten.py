"""Make an inspectable source draft from the retained decode trace (no dispatch)."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import symtable

SOURCE = Path('/Volumes/APDataStore/pact/ddm_pd6/candidate2/candidate_runtime')
ROOT = Path('/Volumes/APDataStore/pact/ddm_mrs1')
NAMES = {
    'cpr1.carrier_codec': 'carrier', 'cpr1.ddm_mp2_semantic_receiver': 'weights',
    'cpr1.hpac_integer': 'prior', 'cpr1.hpac_integer_sparse': 'sparse',
    'cpr1.inflate': 'render', 'cpr1.integer_model_io': 'prior_io',
    'runtime.baseline': 'schema', 'runtime.bits': 'bits',
    'runtime.carrier_repack': 'prediction', 'runtime.compensation_overlay': 'overlay',
    'runtime.dx2_cabac_coefficients': 'coefficients',
    'runtime.entropy.coefficient_ar1_codec': 'trajectory',
    'runtime.entropy.coefficient_predictor': 'predictor',
    'runtime.frame0_selector': 'selector', 'runtime.free_corrector': 'miss',
    'runtime.fx1_logistic_mixer_corrector': 'odds',
    'runtime.fx2_model_axis_corrector': 'context',
    'runtime.hpac_inference': 'inference', 'runtime.ihs2': 'prior_format',
    'runtime.rc1_adaptive_model_sections': 'model_bits',
    'runtime.rc2_hpac_semistatic_mixing': 'adaptive',
    'runtime.rc3_shared_mixer': 'prior_mixer', 'runtime.residual_archive': 'archive',
    'runtime.rlc1_mixer': 'geometry_mixer', 'runtime.rr4_free_corrector': 'counts',
    'runtime.rr5_arith_basis': 'basis', 'runtime.sm1_semantic_mixer': 'weight_mixer',
    'runtime.tc1_shared_mixer': 'shared',
}
KEEP_ALL = {'runtime.free_corrector', 'runtime.fx1_logistic_mixer_corrector',
    'runtime.fx2_model_axis_corrector', 'runtime.rr4_free_corrector', 'cpr1.inflate', 'runtime.frame0_selector'}


def main():
    coverage = json.loads((ROOT / 'trace/COVERAGE.json').read_text())['lines']
    trees, tables, globals_by_module, imports = {}, {}, {}, {}
    for module in NAMES:
        path = SOURCE / (module.replace('.', '/') + '.py')
        source = path.read_text()
        trees[module] = ast.parse(source)
        tables[module] = symtable.symtable(source, str(path), 'exec')
        globals_by_module[module] = {s.get_name() for s in tables[module].get_symbols()
                                     if s.is_assigned() or s.is_imported() or s.is_namespace()}
        imports[module] = {}
        for node in ast.walk(trees[module]):
            if isinstance(node, ast.ImportFrom):
                target = '.'.join(module.split('.')[:-node.level]) if node.level else ''
                if node.module:
                    target = target + '.' + node.module if target else node.module
                if not node.level and target not in NAMES:
                    if 'cpr1.' + target in NAMES:
                        target = 'cpr1.' + target
                    elif target == 'rc1_adaptive_model_sections':
                        target = 'runtime.rc1_adaptive_model_sections'
                for name in node.names:
                    full = target + ('.' if target else '') + name.name
                    if target in NAMES or full in NAMES:
                        key = name.asname or name.name
                        if key not in imports[module] or node.lineno in coverage.get(module.replace('.', '/') + '.py', []):
                            imports[module][key] = (target, name.name)
            elif isinstance(node, ast.Import):
                for name in node.names:
                    if name.name in NAMES:
                        imports[module][name.asname or name.name] = (name.name, None)

    def resolve(module, name, seen=()):
        if (module, name) in seen:
            raise ValueError(('cyclic alias', module, name))
        if name in imports[module]:
            target, member = imports[module][name]
            if member is None or target + '.' + member in NAMES:
                return ('module', target if member is None else target + '.' + member)
            return resolve(target, member, seen + ((module, name),))
        return ('name', NAMES[module] + '_' + name.lstrip('_'))

    class Flatten(ast.NodeTransformer):
        def __init__(self, module):
            self.module = module
            self.scopes = [tables[module]]
            self.hits = set(coverage.get(module.replace('.', '/') + '.py', []))

        def global_name(self, name):
            try:
                symbol = self.scopes[-1].lookup(name)
            except KeyError:
                return True
            return len(self.scopes) == 1 or symbol.is_global()

        def visit_Name(self, node):
            if node.id in imports[self.module] or (self.global_name(node.id) and node.id in globals_by_module[self.module]):
                kind, target = resolve(self.module, node.id)
                if kind == 'name':
                    return ast.copy_location(ast.Name(target, node.ctx), node)
            return node

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id in imports[self.module]:
                kind, target = resolve(self.module, node.value.id)
                if kind == 'module' and node.attr in globals_by_module[target]:
                    kind, target = resolve(target, node.attr)
                    if kind == 'name':
                        return ast.copy_location(ast.Name(target, node.ctx), node)
            return self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if node.module == '__future__':
                return None
            kept = []
            for alias in node.names:
                key = alias.asname or alias.name
                if key not in imports[self.module]:
                    if len(self.scopes) == 1:
                        alias.asname = resolve(self.module, key)[1]
                    kept.append(alias)
            node.names = kept
            return node if kept else None

        def visit_Import(self, node):
            kept = []
            for alias in node.names:
                key = alias.asname or alias.name.split('.')[0]
                if key not in imports[self.module]:
                    if len(self.scopes) == 1:
                        alias.asname = resolve(self.module, key)[1]
                    kept.append(alias)
            node.names = kept
            return node if kept else None

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return None
            return self.generic_visit(node)

        def live(self, body):
            return self.module in KEEP_ALL or any(
                line in self.hits for statement in body
                for line in range(statement.lineno, statement.end_lineno + 1))

        def visit_If(self, node):
            if self.module in KEEP_ALL:
                return self.generic_visit(node)
            if self.live(node.body):
                node.orelse = node.orelse if self.live(node.orelse) else []
                node = self.generic_visit(node)
                if not node.body:
                    node.body = [ast.Pass()]
                return node
            if self.live(node.orelse):
                return [n for old in node.orelse if (n := self.visit(old)) is not None]
            return None

        def visit_Try(self, node):
            if self.module in KEEP_ALL:
                return self.generic_visit(node)
            node.handlers = [h for h in node.handlers if self.live(h.body)]
            node.finalbody = node.finalbody if self.live(node.finalbody) else []
            if not node.handlers and not node.finalbody:
                return [n for old in node.body + node.orelse if (n := self.visit(old)) is not None]
            return self.generic_visit(node)

        def visit_FunctionDef(self, node):
            if node.name in {'snapshot', 'restore'} and self.module in {'runtime.rlc1_mixer', 'runtime.tc1_shared_mixer'}:
                return None
            if self.module not in KEEP_ALL and not any(
                line in self.hits for statement in node.body
                for line in range(statement.lineno, statement.end_lineno + 1)
            ):
                return None
            old_name = node.name
            if len(self.scopes) == 1:
                node.name = resolve(self.module, old_name)[1]
            node.decorator_list = [self.visit(d) for d in node.decorator_list]
            node.args.defaults = [self.visit(d) for d in node.args.defaults]
            node.args.kw_defaults = [self.visit(d) if d is not None else None for d in node.args.kw_defaults]
            # Annotations document the original modular source; omit them in this draft.
            node.returns = None
            for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                a.annotation = None
            if node.args.vararg:
                node.args.vararg.annotation = None
            if node.args.kwarg:
                node.args.kwarg.annotation = None
            children = [s for s in self.scopes[-1].get_children() if s.get_name() == old_name and s.get_lineno() == node.lineno]
            self.scopes.append(children[0])
            body = []
            for old in node.body:
                new = self.visit(old)
                body.extend(new if isinstance(new, list) else [new] if new is not None else [])
            node.body = body
            self.scopes.pop()
            if not node.body:
                node.body = [ast.Pass()]
            return node

        def visit_ClassDef(self, node):
            old_name = node.name
            node.name = resolve(self.module, old_name)[1]
            node.bases = [self.visit(b) for b in node.bases]
            node.decorator_list = [self.visit(d) for d in node.decorator_list]
            children = [s for s in self.scopes[-1].get_children() if s.get_name() == old_name and s.get_lineno() == node.lineno]
            self.scopes.append(children[0])
            body = []
            for old in node.body:
                new = self.visit(old)
                body.extend(new if isinstance(new, list) else [new] if new is not None else [])
            node.body = body
            self.scopes.pop()
            if not node.body:
                node.body = [ast.Pass()]
            return node

    order, seen = [], set()
    def visit(module):
        if module in seen:
            return
        seen.add(module)
        for target, member in imports[module].values():
            visit(target + '.' + member if member is not None and target + '.' + member in NAMES else target)
        order.append(module)
    for module in NAMES:
        visit(module)
    sections = []
    for module in order:
        tree = Flatten(module).visit(copy.deepcopy(trees[module]))
        ast.fix_missing_locations(tree)
        sections.append('# ' + NAMES[module].replace('_', ' ').capitalize() + '\n' + ast.unparse(tree))
    draft = 'from __future__ import annotations\n\n' + '\n\n'.join(sections) + '\n'
    draft = draft.replace("integer = prior_format_importlib.import_module('hpac_integer')", 'integer = prior_namespace')
    draft = draft.replace('model_module = inference_sys.modules[type(model).__module__]', 'model_module = prior_namespace')
    draft = draft.replace('model_module.ste_round = inference_inference_round', 'global prior_ste_round, prior_requantize\n    prior_ste_round = inference_inference_round')
    draft = draft.replace('model_module.requantize = inference_inference_requantize', 'prior_requantize = inference_inference_requantize')
    draft = draft.replace('from .rlc1_geometry import FORMAT as geometry_mixer_FORMAT, LaneGeometry as geometry_mixer_LaneGeometry, parse_config as geometry_mixer_parse_config', 'geometry_mixer_FORMAT = struct.Struct("<BHH8B6B")\ngeometry_mixer_LaneGeometry = CausalGeometry\ngeometry_mixer_parse_config = parse_geometry_config')
    draft = draft.replace('from .entropy.rc64 import NativeDecoder as archive_NativeDecoder', '')
    draft = '\n'.join(line for line in draft.splitlines() if not line.startswith('from .entropy.renderer_weight_codec import ')) + '\n'
    primitive_path = Path('experiments/ddm_mrs1_python_primitives.py')
    primitives = primitive_path.read_text().replace('from __future__ import annotations', '')
    source = primitives + '\n' + draft.replace('from __future__ import annotations', '')
    source += '\n' + Path('experiments/ddm_mrs1_receiver_main.inc').read_text()
    tree = ast.parse(source)
    def defines(node):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            return {node.name}
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return {a.asname or a.name.split('.')[0] for a in node.names}
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            return {n.id for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
                    for n in ast.walk(t) if isinstance(n, ast.Name)}
        return set()
    names = set().union(*(defines(n) for n in tree.body))
    needed = {'main', 'TokenDecoder', 'read_models', 'write_video'}
    # The explicit renderer namespace is a compatibility seam for the inherited parser.
    needed |= {n for n in names if n.startswith('render_') and not n.startswith('render_unpack')}
    while True:
        before = needed.copy()
        for node in tree.body:
            if defines(node) & needed or (isinstance(node, ast.Expr) and
                    {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} & needed):
                needed |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        if needed == before:
            break
    tree.body = [n for n in tree.body if defines(n) & needed or
                 (isinstance(n, ast.Expr) and {v.id for v in ast.walk(n) if isinstance(v, ast.Name)} & needed) or
                 (isinstance(n, ast.If) and '__name__' in ast.unparse(n.test))]
    path = ROOT / 'inflate_draft.py'
    path.write_text('"""Decode the counted class field, token renderer, and pose carrier."""\n'
                    + 'from __future__ import annotations\n\n' + ast.unparse(tree) + '\n')
    print(json.dumps(dict(path=str(path), lines=len(path.read_text().splitlines()))))


if __name__ == '__main__':
    main()
